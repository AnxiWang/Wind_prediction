import netCDF4 as nc
import numpy as np
import pandas as pd
import tqdm
import os
import datetime as dt
import glob

ymdh = "%Y%m%d%H"


def ymd(str):
    return pd.to_datetime(str)


def ymd_hms(str):
    return pd.to_datetime(str)


def hours(h):
    return dt.timedelta(hours=h)


def ymd_h(x):
    return pd.to_datetime(x, format="%Y%m%d%H")


def as_str(x):
    return x.strftime("%Y%m%d")


def get_gts_time(gts_dir):
    import glob
    files = glob.glob("{0}/*.csv".format(gts_dir))
    times = ymd([x[-17:-9] for x in files])
    return times.sort_values()


def get_wrf_time(wrf_dir):
    fileNames = os.listdir(wrf_dir)
    dirs = []
    for name in fileNames:
        if len(name) > 35:
            dirs.append(name.split('.')[-2].split('_')[0])
    # dirs = [name.split('.')[-2].split('_')[0] for name in os.listdir(wrf_dir)]
    return pd.to_datetime(dirs, format=ymdh)


def load_gts(gts_dir, gts_times):
    gts_files = ["{0}/GTS.out_{1}_wind.csv".format(gts_dir, as_str(x)) for x in gts_times]
    res = [pd.read_csv(f, index_col=0) for f in gts_files]
    total = pd.concat(res)
    # convert all float64 type to float32
    for c in total.columns:
        if total[c].dtype.name == 'float64':
            total[c] = total[c].astype('float32')
        if total[c].dtype.name == 'int64':
            total[c] = total[c].astype('int32')
    total.reset_index(inplace=True)
    total['Time'] = ymd_h(total['Time'])
    total['stationID'] = total['stationID'].astype('str')
    return total


def remove_duplicate_gts(gts):
    x = gts[['stationID', 'Time', 'LONG']].groupby(['stationID', 'Time']).count()
    # print(x)
    tmp = x[x['LONG'] > 1].reset_index()
    # 目前先不考虑去重，数据中已经发现存在同一站点同一时间存在多个观测值
    dc = tmp['LONG'].count()
    print("{0} duplicated records have been found and removed".format(dc))
    x = gts.groupby(['stationID', 'Time']).first().reset_index()
    return x


# 删除一行数据中超过两个空值的行
def remove_abnormal(gts):
    gts = gts.dropna(thresh=2)
    return gts


def dump_wrf_var(wrf_dir, spot, pred):
    spot_str = spot.strftime(ymdh)
    pred_str = pred.strftime(ymdh)
    res = []
    for i in range(0, len(spot_str)):
        ncName = 'pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
        # file = sorted(glob.glob(wrf_dir + '*pwrfout_d01.' + spot_str[i] + '*.nc'))
        file = wrf_dir + ncName
        print(file)

        with nc.Dataset(file, mode='r', format='NETCDF4_CLASSIC') as ds:
            # print(len(ds['U10'][:].data[0]))    # 299
            # print(len(ds['V10'][:].data))       # 133
            # print([spot_str[i], pred_str[i], ds['U10'][:].data, ds['V10'][:].data])
            res.append([spot_str[i], pred_str[i], ds['U10'][:].data, ds['V10'][:].data])

    return res

    #     filelist.extend(file)
    # print(filelist)
    # 三种命名方式的文件，需要分别加入读取列表
    # filelist = []
    # filelist.extend(sorted(glob.glob(wrf_dir + '6h.pack.pwrfout_d01.' + str(yy) + '*.nc')))
    # filelist.extend(sorted(glob.glob(wrf_dir + 'pack.pwrfout_d01.' + str(yy) + '*.nc')))
    # filelist.extend(sorted(glob.glob(wrf_dir + 'pwrfout_d01.' + str(yy) + '*.nc')))
    # nt = len(filelist)
    # files = [wrf_dir + '/' + r'\.' + 'pwrfout_d01.{0}_{1}.nc'.format(x, y)
    #          for x, y in zip(spot_str, pred_str)]
    # print(files)
    # res = []

    # for s, p, f in tqdm(zip(spot, pred, files)):
    #     print(s + ' ' + p + ' ' + f)
    #     if not os.path.exists(f):
    #         print("Warning: {0} does not exists.".format(f))
    #         continue
    #     try:
    #         with Dataset(f) as ds:
    #             res.append([s, p, ds[var][:].data])
    #     except:
    #         print("Warning: error in reading {0} from {1}.".format(var, f))
    #         continue
    # return res


# build mesh (stored in DataFrame) for EC data：为EC数据建立网格
def build_mesh():
    ds = nc.Dataset('../../data/wrfout/pwrfout_d01.2019010100_2019010812.nc')
    lon = ds['XLONG'][:].data
    lat = ds['XLAT'][:].data
    nlon = len(lon[0][0])
    nlat = len(lat[0][:, 0])
    idx = np.meshgrid(np.array(range(0, nlon)), np.array(range(0, nlat)))
    d = np.meshgrid(lon[0][0], lat[0][:, 0])
    return pd.DataFrame({'ilon': idx[0].ravel(), 'ilat': idx[1].ravel(),
                         'lon': d[0].ravel(), 'lat': d[1].ravel()})


def find_nearest_point(p, grid):
    grid['dist'] = (grid['lon'] - p[0]) ** 2 + (grid['lat'] - p[1]) ** 2
    return grid.nsmallest(4, columns='dist')


def tp_mesh_to_station(obs, tp_pred, mesh):
    stations = obs.groupby('Station_Id').first().reset_index() \
        [['Station_Id', 'Longitude', 'Latitude']]

    res = []
    for index, row in tqdm(stations.iterrows(), total=stations.shape[0]):
        s_lon = row['Longitude']
        s_lat = row['Latitude']
        nn = find_nearest_point([s_lon, s_lat], mesh)
        # caluate the weight for each point
        # 选取距离圆形中心最近四个EC网格数据，按照距离站点的远近赋予权重，
        # 距离越近权重越大，得到一个插值结果，来辅助站点观测数据，进行模型预测
        nn['weight'] = (1 / nn["dist"]) / sum(1 / nn["dist"])
        ilon = np.array(nn['ilon'], dtype=int)
        ilat = np.array(nn['ilat'], dtype=int)
        weight = np.array(nn['weight'])

        t1 = np.array([x[0] for x in tp_pred])
        t2 = np.array([x[1] for x in tp_pred])

        v0 = np.array([x[2][0, ilat[0], ilon[0]] for x in tp_pred])
        v1 = np.array([x[2][0, ilat[1], ilon[1]] for x in tp_pred])
        v2 = np.array([x[2][0, ilat[2], ilon[2]] for x in tp_pred])
        v3 = np.array([x[2][0, ilat[3], ilon[3]] for x in tp_pred])
        # v = v0
        # print(max(v))
        v = v0 * weight[0] + v1 * weight[1] + v2 * weight[2] + v3 * weight[3]

        df = pd.DataFrame({'Station_Id': row['Station_Id'],
                           'SpotTime': t1,
                           'PredTime': t2,
                           'PredTP': v * 1000})

        res.append(df)
    return pd.concat(res).reset_index().drop(columns=['index'])