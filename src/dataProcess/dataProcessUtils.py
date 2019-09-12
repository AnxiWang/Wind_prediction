import time

import netCDF4 as nc
import numpy as np
import pandas as pd
import tqdm
import os
import datetime as dt
import math
import glob

ymdh = "%Y%m%d%H"
u10_col = []
v10_col = []
for col_i in range(0, 21, 1):
    u10_col.append(str(col_i * 6) + 'h_u10')
    v10_col.append(str(col_i * 6) + 'h_v10')
u10_col_1h = []
v10_col_1h = []
for col_i in range(0, 133, 1):
    u10_col_1h.append(str(col_i) + 'h_u10')
    v10_col_1h.append(str(col_i) + 'h_v10')


def ymd(str):
    return pd.to_datetime(str)


def ymd_hms(str):
    return pd.to_datetime(str)


def hours(h):
    return dt.timedelta(hours=h)


def ym(x):
    return pd.to_datetime(x, format="%Y%m")


def ymd_h(x):
    return pd.to_datetime(x, format="%Y%m%d%H")


def as_str(x):
    return x.strftime("%Y%m%d")


def as_str_h(x):
    return x.strftime("%Y%m%d%H")


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


def is_valid_date(str):
  '''判断是否是一个有效的日期字符串'''
  try:
    ymd_h(str)
    return True
  except:
    return False


def load_gts(gts_dir, gts_times):
    gts_files = ["{0}/GTS.out_{1}_wind.csv".format(gts_dir, as_str(x)) for x in gts_times]
    res = [pd.read_csv(f, index_col=0, encoding='windows-1252') for f in gts_files]
    if res:     # 月份观测数据缺失导致res为空列表
        total = pd.concat(res)
        # convert all float64 type to float32
        for c in total.columns:
            if total[c].dtype.name == 'float64':
                total[c] = total[c].astype('float32')
            if total[c].dtype.name == 'int64':
                total[c] = total[c].astype('int32')
        total.reset_index(inplace=True)
        # total['Time'] = ymd_h(total['Time'])

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


# dump wrf data into one (2013-2017 by year)
def dump_wrf_var(wrf_dir, spot, pred, year):
    spot_str = spot.strftime(ymdh)
    pred_str = pred.strftime(ymdh)
    res = []
    for i in range(0, len(spot_str)):
        if spot_str[i][0:4] == year:
            ncName = '6h.pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
            # file = sorted(glob.glob(wrf_dir + '*pwrfout_d01.' + spot_str[i] + '*.nc'))
            file = wrf_dir + ncName
            print(file)
            with nc.Dataset(file, mode='r', format='NETCDF4_CLASSIC') as ds:
                variable_keys = ds.variables.keys()
                # print(variable_keys)
                if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys and 'XLAT' in variable_keys:
                    res.append([(ymd_h(spot_str[i]) + hours(12)).strftime(ymdh),
                                (ymd_h(pred_str[i]) - hours(2 * 24)).strftime(ymdh),
                                ds['XLONG'][:].data,
                                ds['XLAT'][:].data,
                                ds['U10'][:].data,
                                ds['V10'][:].data])
    return res


# dump pcwrf data into one (2013-2017 by year)
def dump_pcwrf_var(near_mesh, wrf_dir, spot, pred, year):
    spot_str = spot.strftime(ymdh)
    pred_str = pred.strftime(ymdh)
    res = []
    for index, var in near_mesh.iterrows():
        print(var['mesh_lon'], var['mesh_lat'])
        for i in range(0, len(spot_str)):
            if spot_str[i][0:4] == year:
                if int(year) < 2017:
                    ncName = '6h.pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                elif int(year) >= 2017:
                    ncName = 'pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                file = wrf_dir + ncName
                print(file)
                with nc.Dataset(file, mode='r', format='NETCDF4_CLASSIC') as ds:
                    variable_keys = ds.variables.keys()
                    if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys and 'XLAT' in variable_keys:

                        # print(ds.variables['U10'][:, var['mesh_lat'], var['mesh_lon']])
                        # # if ds['XLONG'][:].data in near_mesh.mesh_lon and ds['XLAT'][:].data in near_mesh.mesh_lat:
                        # print(type(ds['XLONG'][:].data))
                        # print(ds['XLONG'][:].data)
                        # exit()
                        res.append([(ymd_h(spot_str[i])).strftime(ymdh),
                                    (ymd_h(pred_str[i])).strftime(ymdh),
                                    var['mesh_lat'],
                                    var['mesh_lon'],
                                    ds.variables['U10'][:, var['mesh_lat'], var['mesh_lon']],
                                    ds.variables['V10'][:, var['mesh_lat'], var['mesh_lon']]])
    return res


# dump wrf data into one (2018-2019 by month)
def dump_wrf_var_month(wrf_dir, spot, pred, month, year):
    spot_str = spot.strftime(ymdh)
    pred_str = pred.strftime(ymdh)
    res = []
    for i in range(0, len(spot_str)):
        if spot_str[i][0:4] == year and spot_str[i][4:6] == month:
            if year == '2018':
                ncName = 'pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
            else:
                ncName = 'pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
            # file = sorted(glob.glob(wrf_dir + '*pwrfout_d01.' + spot_str[i] + '*.nc'))
            file = wrf_dir + ncName
            print(file)
            with nc.Dataset(file, mode='r', format='NETCDF4_CLASSIC') as ds:
                variable_keys = ds.variables.keys()
                # print(variable_keys)
                if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys and 'XLAT' in variable_keys:
                    res.append([(ymd_h(spot_str[i]) + hours(12)).strftime(ymdh),
                                (ymd_h(pred_str[i]) - hours(2 * 24)).strftime(ymdh),
                                ds['XLONG'][:].data,
                                ds['XLAT'][:].data,
                                ds['U10'][:].data,
                                ds['V10'][:].data])
    return res


# dump pcwrf data into one (2017-2018 by month)
def dump_pcwrf_var_month(near_mesh, wrf_dir, spot, pred, month, year):
    spot_str = spot.strftime(ymdh)
    pred_str = pred.strftime(ymdh)
    res = []
    for i in range(0, len(spot_str)):
        if spot_str[i][0:4] == year and spot_str[i][4:6] == month:
            # if year == '2017':
            #     ncName = 'pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
            # else:
            ncName = 'pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
            file = wrf_dir + ncName
            print(file)
            with nc.Dataset(file, mode='r', format='NETCDF4_CLASSIC') as ds:
                variable_keys = ds.variables.keys()
                if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys and 'XLAT' in variable_keys:
                    res.append([(ymd_h(spot_str[i])).strftime(ymdh),
                                (ymd_h(pred_str[i])).strftime(ymdh),
                                ds['XLONG'][:].data,
                                ds['XLAT'][:].data,
                                ds['U10'][:].data,
                                ds['V10'][:].data])
    return res


# build mesh (stored in DataFrame) for EC data
def build_mesh(mesh):
    ds = nc.Dataset(mesh)
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


def find_mesh_city_station(station, mesh):
    res = pd.DataFrame()
    for index in range(0, station.shape[0], 1):
        s_lon = station.loc[index].LONG
        s_lat = station.loc[index].LAT
        nn = find_nearest_point([s_lon, s_lat], mesh)
        nn = nn.rename(columns={'lon': 'mesh_lon', 'lat': 'mesh_lat'})
        col_name = nn.columns.tolist()  # 将数据框的列名全部提取出来存放在列表里
        col_name.insert(1, 'sta_lon')
        col_name.insert(2, 'sta_lat')
        nn = nn.reindex(columns=col_name)  # DataFrame.reindex() 对原行/列索引重新构建索引值
        nn['sta_lon'] = s_lon  # 给city列赋值
        nn['sta_lat'] = s_lat
        res = res.append(nn)
    return res


def wrf_mesh_to_station(station, wrf_pred, mesh):
    res = []
    ncNumber = len(wrf_pred[0])
    # print(wrf_pred[0][0][0], wrf_pred[0][0][1], len(wrf_pred[0][0][2]), len(wrf_pred[0][0][3]), len(wrf_pred[0][0][4]),len(wrf_pred[0][0][5]))     # len(wrf_pred[0][0]) = 6 len(wrf_pred[0]) = 5
    # print(wrf_pred[0][1][0], wrf_pred[0][1][1], len(wrf_pred[0][1][2]))
    # print(wrf_pred[0][2][0], wrf_pred[0][2][1], len(wrf_pred[0][2][2]))
    # print(wrf_pred[0][3][0], wrf_pred[0][3][1])
    # print(wrf_pred[0][4][0], wrf_pred[0][4][1])
    # # c = {"spot_time": wrf_pred[0][0][: :-1],
    # #      "pred_time": wrf_pred[0][1][: :-1],
    # #      "XLONG": wrf_pred[0][2]}
    # # data = pd.DataFrame(c)  # 将字典转换成为数据框
    # print(len(wrf_pred[0][0][0][0]))
    # print(len(wrf_pred[0][0][1][0]))
    # print(len(wrf_pred[0][0][2][0][0]))
    # print(len(wrf_pred[0][0][3][0][0]))
    # print(len(wrf_pred[0][0][4][0][0]))
    # print(len(wrf_pred[0][0][5][0]))
    # 2013040900
    # 2013041400
    # 21
    # 21
    # 21
    # 21
    # 2013040912
    # 2013041412
    # 21
    # 2013041000
    # 2013041500
    # 21
    # 2013041012
    # 2013041512
    # 2013041100
    # 2013041600
    # 1
    # 1
    # 549
    # 549
    # 549
    # 299

    for index in range(0, station.shape[0], 1):
        s_lon = station.loc[index].LONG
        s_lat = station.loc[index].LAT
        nn = find_nearest_point([s_lon, s_lat], mesh)
        # caluate the weight for each point
        # 选取距离圆形中心最近四个EC网格数据，按照距离站点的远近赋予权重，
        # 距离越近权重越大，得到一个插值结果，来辅助站点观测数据，进行模型预测
        nn['weight'] = (1 / nn["dist"]) / sum(1 / nn["dist"])
        ilon = np.array(nn['ilon'], dtype=int)
        ilat = np.array(nn['ilat'], dtype=int)
        weight = np.array(nn['weight'])

        for eachNc in range(0, ncNumber, 1):
            t1 = np.array([x[eachNc][0] for x in wrf_pred])
            t2 = np.array([x[eachNc][1] for x in wrf_pred])
            print(t1, t2)
            # 取出的u0，u1，u2，u3都是一个21维的数组
            u0 = np.array([x[eachNc][4][:, ilat[0], ilon[0]] for x in wrf_pred])
            u1 = np.array([x[eachNc][4][:, ilat[1], ilon[1]] for x in wrf_pred])
            u2 = np.array([x[eachNc][4][:, ilat[2], ilon[2]] for x in wrf_pred])
            u3 = np.array([x[eachNc][4][:, ilat[3], ilon[3]] for x in wrf_pred])
            # print(u0, u1, u2, u3)

            v0 = np.array([x[eachNc][5][:, ilat[0], ilon[0]] for x in wrf_pred])
            v1 = np.array([x[eachNc][5][:, ilat[1], ilon[1]] for x in wrf_pred])
            v2 = np.array([x[eachNc][5][:, ilat[2], ilon[2]] for x in wrf_pred])
            v3 = np.array([x[eachNc][5][:, ilat[3], ilon[3]] for x in wrf_pred])
            # print(v0, v1, v2, v3)
            u = u0 * weight[0] + u1 * weight[1] + u2 * weight[2] + u3 * weight[3]
            v = v0 * weight[0] + v1 * weight[1] + v2 * weight[2] + v3 * weight[3]
            # print(u[0], v[0])
            df = pd.DataFrame(
                {'Station_Id': station.loc[index].stationID, 'XLONG': s_lon, 'XLAT': s_lat, 'SpotTime': t1,
                 'PredEndTime': t2})

            if len(u[0]) == 21:
                u10_pd = pd.DataFrame(u, columns=u10_col)
                v10_pd = pd.DataFrame(v, columns=v10_col)
                df = df.join(u10_pd, how='right')
                df = df.join(v10_pd, how='right')
                res.append(df)
            elif len(u[0]) < 21:
                u = u.tolist()
                v = v.tolist()
                u[0].extend(np.nan for _ in range(21 - len(u[0])))
                v[0].extend(np.nan for _ in range(21 - len(v[0])))

                u10_pd = pd.DataFrame(u, columns=u10_col)
                v10_pd = pd.DataFrame(v, columns=v10_col)
                df = df.join(u10_pd, how='right')
                df = df.join(v10_pd, how='right')
                res.append(df)
            elif 21 < len(u[0]) <= 133:
                u = u.tolist()
                v = v.tolist()
                u[0].extend(np.nan for _ in range(133 - len(u[0])))
                v[0].extend(np.nan for _ in range(133 - len(v[0])))

                u10_pd_1h = pd.DataFrame(u, columns=u10_col_1h)
                v10_pd_1h = pd.DataFrame(v, columns=v10_col_1h)
                df = df.join(u10_pd_1h, how='right')
                df = df.join(v10_pd_1h, how='right')
                res.append(df)
            else:
                print('error!')
    return pd.concat(res).reset_index().drop(columns=['index'])


def compute(u, v):
    if u > 0 and v > 0:
        fx = 270 - math.atan(v / u) * 180 / math.pi
    elif u < 0 and v > 0:
        fx = 90 - math.atan(v / u) * 180 / math.pi
    elif u < 0 and v < 0:
        fx = 90 - math.atan(v / u) * 180 / math.pi
    elif u > 0 and v < 0:
        fx = 270 - math.atan(v / u) * 180 / math.pi
    elif u == 0 and v > 0:
        fx = 180
    elif u == 0 and v < 0:
        fx = 0
    elif u > 0 and v == 0:
        fx = 270
    elif u < 0 and v == 0:
        fx = 90
    elif u == 0 and v == 0:
        fx = 999.9
    elif pd.isnull(u) and pd.isnull(v):
        fx = np.nan

    # 风速是uv分量的平方和
    fs = math.sqrt(math.pow(u, 2) + math.pow(v, 2))
    return fx, fs


def calculateSpeedFromUV(sta_wrf_pred):
    afterDf = pd.DataFrame(columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime', 'Direction', 'Speed'])

    for index, row in sta_wrf_pred.iterrows():
        if len(row) == 47:
            for x, y in zip(u10_col, v10_col):
                predTime = ymd_h(row['SpotTime']) + hours(int(x.split('h')[0]))
                predTimeStr = as_str_h(predTime)
                direction, speed = compute(row[x], row[y])
                print(row['Station_Id'], row['XLONG'], row['XLAT'], row['SpotTime'], predTimeStr, direction, speed)
                afterDf = afterDf.append([{'Station_Id': row.Station_Id,
                                   'XLONG': row.XLONG,
                                   'XLAT': row.XLAT,
                                   'SpotTime': row.SpotTime,
                                   'PredTime': predTimeStr,
                                   'Direction': direction,
                                   'Speed': speed
                                           }], ignore_index=True)
        elif len(row) == 271:
            for x, y in zip(u10_col_1h, v10_col_1h):
                predTime = ymd_h(row['SpotTime']) + hours(int(x.split('h')[0]))
                predTimeStr = as_str_h(predTime)
                direction, speed = compute(row[x], row[y])
                print(row['Station_Id'], row['XLONG'], row['XLAT'], row['SpotTime'], predTimeStr, direction, speed)
                afterDf = afterDf.append([{'Station_Id': row.Station_Id,
                                   'XLONG': row.XLONG,
                                   'XLAT': row.XLAT,
                                   'SpotTime': row.SpotTime,
                                   'PredTime': predTimeStr,
                                   'Direction': direction,
                                   'Speed': speed
                                           }], ignore_index=True)
    return afterDf
