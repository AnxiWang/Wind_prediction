import time

import netCDF4 as nc
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import datetime as dt
import math
import glob
import re
import warnings

warnings.filterwarnings("ignore")

ymdh = "%Y%m%d%H"
u10_col = []
v10_col = []
for col_i in range(0, 21, 1):
    u10_col.append(str(col_i * 6) + 'h')
    v10_col.append(str(col_i * 6) + 'h_v10')
u10_col_1h = []
v10_col_1h = []
for col_i in range(0, 133, 1):
    u10_col_1h.append(str(col_i) + 'h')
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
    if res:  # 月份观测数据缺失导致res为空列表
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
def dump_wrf_var(near_mesh, wrf_dir, spot, pred, year):
    global nc_name
    spot_str = spot.strftime(ymdh)
    pred_str = pred.strftime(ymdh)
    res = []
    for index, var in tqdm(near_mesh.iterrows(), total=near_mesh.shape[0]):
        for i in tqdm(range(0, len(spot_str)), total=len(spot_str)):
            if spot_str[i][0:4] == year:
                if int(year) < 2018:
                    nc_name = '6h.pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                elif int(year) == 2018:
                    nc_name = 'pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                elif int(year) > 2018:
                    nc_name = 'pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                file = wrf_dir + nc_name
                # print(file)
                with nc.Dataset(file, mode='r', format='NETCDF4_CLASSIC') as ds:
                    variable_keys = ds.variables.keys()
                    # print(variable_keys)
                    if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys \
                            and 'XLAT' in variable_keys and 'UU' in variable_keys and 'VV' in variable_keys \
                            and 'Q2' in variable_keys and 'PSFC' in variable_keys and 'QVAPOR' in variable_keys \
                            and 'QCLOUD' in variable_keys and 'HGT' in variable_keys and 'RAINC' in variable_keys \
                            and 'RAINNC' in variable_keys and 'SWDOWN' in variable_keys and 'GSW' in variable_keys \
                            and 'GLW' in variable_keys and 'HFX' in variable_keys and 'QFX' in variable_keys \
                            and 'LH' in variable_keys and 'TT' in variable_keys and 'GHT' in variable_keys \
                            and 'RH' in variable_keys and 'SLP' in variable_keys:
                        xlon = ds.variables['XLONG'][:][0][0].tolist()
                        if 28 < min(xlon) and max(xlon) < 132:
                            res.append([(ymd_h(spot_str[i]) + hours(12)).strftime(ymdh),
                                        (ymd_h(pred_str[i]) - hours(2 * 24)).strftime(ymdh),
                                        var['mesh_lat'],
                                        var['mesh_lon'],
                                        ds.variables['U10'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['V10'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['UU'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['VV'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['Q2'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['T2'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['PSFC'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['QVAPOR'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['QCLOUD'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['HGT'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['RAINC'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['RAINNC'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['SWDOWN'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['GSW'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['GLW'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['HFX'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['QFX'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['LH'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['TT'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['GHT'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['RH'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['SLP'][:, var['mesh_lat'], var['mesh_lon']]
                                        ])
                    # if 'UU' in variable_keys and 'VV' in variable_keys:
                    #     res.append([ds.variables['UU'][:, 0, var['mesh_lat'], var['mesh_lon']],
                    #                 ds.variables['VV'][:, 0, var['mesh_lat'], var['mesh_lon']]])
                    # if 'Q2' in variable_keys:
                    #     res.append(ds.variables['Q2'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'T2' in variable_keys:
                    #     res.append(ds.variables['T2'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'PSFC' in variable_keys:
                    #     res.append(ds.variables['PSFC'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'QVAPOR' in variable_keys:
                    #     res.append(ds.variables['QVAPOR'][:, 0, var['mesh_lat'], var['mesh_lon']])
                    # if 'QCLOUD' in variable_keys:
                    #     res.append(ds.variables['QCLOUD'][:, 0, var['mesh_lat'], var['mesh_lon']])
                    # if 'HGT' in variable_keys:
                    #     res.append(ds.variables['HGT'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'RAINC' in variable_keys:
                    #     res.append(ds.variables['RAINC'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'RAINNC' in variable_keys:
                    #     res.append(ds.variables['RAINNC'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'SWDOWN' in variable_keys:
                    #     res.append(ds.variables['SWDOWN'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'GSW' in variable_keys:
                    #     res.append(ds.variables['GSW'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'GLW' in variable_keys:
                    #     res.append(ds.variables['GLW'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'HFX' in variable_keys:
                    #     res.append(ds.variables['HFX'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'QFX' in variable_keys:
                    #     res.append(ds.variables['QFX'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'LH' in variable_keys:
                    #     res.append(ds.variables['LH'][:, var['mesh_lat'], var['mesh_lon']])
                    # if 'TT' in variable_keys:
                    #     res.append(ds.variables['TT'][:, 0, var['mesh_lat'], var['mesh_lon']])
                    # if 'GHT' in variable_keys:
                    #     res.append(ds.variables['GHT'][:, 0, var['mesh_lat'], var['mesh_lon']])
                    # if 'RH' in variable_keys:
                    #     res.append(ds.variables['RH'][:, 0, var['mesh_lat'], var['mesh_lon']])
                    # if 'SLP' in variable_keys:
                    #     res.append(ds.variables['SLP'][:, var['mesh_lat'], var['mesh_lon']])
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
    nc_num = int(ncNumber / 172)  # 502

    for index in range(0, station.shape[0], 1):
        s_lon = station.loc[index].LONG
        s_lat = station.loc[index].LAT
        # find wrf values of four mesh points(for one station)
        mesh_00 = [wrf_pred[0][i] for i in range((4 * index + 0) * nc_num, (4 * index + 1) * nc_num, 1)]
        mesh_01 = [wrf_pred[0][i] for i in range((4 * index + 1) * nc_num, (4 * index + 2) * nc_num, 1)]
        mesh_02 = [wrf_pred[0][i] for i in range((4 * index + 2) * nc_num, (4 * index + 3) * nc_num, 1)]
        mesh_03 = [wrf_pred[0][i] for i in range((4 * index + 3) * nc_num, (4 * index + 4) * nc_num, 1)]
        nn = find_nearest_point([s_lon, s_lat], mesh)
        nn['weight'] = (1 / nn["dist"]) / sum(1 / nn["dist"])
        weight = np.array(nn['weight'])
        # each_res = pd.DataFrame()
        for each_nc in range(0, nc_num, 1):
            t1_0 = mesh_00[each_nc][0]
            t2_0 = mesh_00[each_nc][1]
            df = pd.DataFrame({'Station_Id': station.loc[index].stationID, 'XLONG': s_lon, 'XLAT': s_lat, 'SpotTime': t1_0,
                 'PredEndTime': t2_0}, index=[0])
            # 22 variables in wrf output file
            for var in range(4, 26, 1):
                var_0 = mesh_00[each_nc][var]
                var_1 = mesh_01[each_nc][var]
                var_2 = mesh_02[each_nc][var]
                var_3 = mesh_03[each_nc][var]

                key = var_0 * weight[0] + var_1 * weight[1] + var_2 * weight[2] + var_3 * weight[3]

                if len(key) == 21:
                    key_pd = pd.DataFrame([key], columns=[x + '_' + str(var) for x in u10_col])
                    df = df.join(key_pd, how='right')
                elif len(key) < 21:
                    key = key.tolist()
                    key.extend(np.nan for _ in range(21 - len(key)))
                    key_pd = pd.DataFrame([key], columns=[x + '_' + str(var) for x in u10_col])
                    df = df.join(key_pd, how='right')
                elif 21 < len(key) <= 133:
                    key = key.tolist()
                    key.extend(np.nan for _ in range(133 - len(key)))
                    key_pd_1h = pd.DataFrame([key], columns=[x + '_' + str(var) for x in u10_col_1h])
                    df = df.join(key_pd_1h, how='right')
                else:
                    print('error!')
            res.append(df)
        # res.append(each_res)
    return pd.concat(res, sort=False).reset_index().drop(columns=['index'])


def compute(u, v):
    global fx
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
    # afterDf = pd.DataFrame(columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime', 'Direction', 'Speed'])
    after_df = pd.DataFrame(columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime', 'Direction', 'Speed', 'UU',
                                     'VV', 'Q2', 'T2', 'PSFC', 'QVAPOR', 'QCLOUD', 'HGT', 'RAINC', 'RAINNC', 'SWDOWN',
                                     'GSW', 'GLW', 'HFX', 'QFX', 'LH', 'TT', 'GHT', 'RH', 'SLP'])

    col_names = sta_wrf_pred.columns.values.tolist()
    u10_columns = [x for x in col_names if re.search('h_4', x)]
    v10_columns = [x for x in col_names if re.search('h_5', x)]
    # 按行遍历wrf的输出结果
    for index, row in sta_wrf_pred.iterrows():
        for x, y in zip(u10_columns, v10_columns):
            hour = x.split('h')[0]
            if int(hour) % 6 == 0:
                pred_time = ymd_h(row['SpotTime']) + hours(int(hour))
                pred_time_str = as_str_h(pred_time)
                direction, speed = compute(row[x], row[y])
                # print(row['Station_Id'], row['XLONG'], row['XLAT'], row['SpotTime'], pred_time_str, direction, speed)
                after_df = after_df.append([{'Station_Id': row.Station_Id,
                                             'XLONG': row.XLONG,
                                             'XLAT': row.XLAT,
                                             'SpotTime': row.SpotTime,
                                             'PredTime': pred_time_str,
                                             'Direction': direction,
                                             'Speed': speed,
                                             'UU': row[hour + 'h_6'],
                                             'VV': row[hour + 'h_7'],
                                             'Q2': row[hour + 'h_8'],
                                             'T2': row[hour + 'h_9'],
                                             'PSFC': row[hour + 'h_10'],
                                             'QVAPOR': row[hour + 'h_11'],
                                             'QCLOUD': row[hour + 'h_12'],
                                             'HGT': row[hour + 'h_13'],
                                             'RAINC': row[hour + 'h_14'],
                                             'RAINNC': row[hour + 'h_15'],
                                             'SWDOWN': row[hour + 'h_16'],
                                             'GSW': row[hour + 'h_17'],
                                             'GLW': row[hour + 'h_18'],
                                             'HFX': row[hour + 'h_19'],
                                             'QFX': row[hour + 'h_20'],
                                             'LH': row[hour + 'h_21'],
                                             'TT': row[hour + 'h_22'],
                                             'GHT': row[hour + 'h_23'],
                                             'RH': row[hour + 'h_24'],
                                             'SLP': row[hour + 'h_25']
                                             }], ignore_index=True)
    return after_df
    #
    # # wrf output per six hours
    # if len(row) == 467:
    #     for x, y in zip(u10_columns, v10_columns):
    #         hour = x.split('h')[0]
    #         pred_time = ymd_h(row['SpotTime']) + hours(int(hour))
    #         pred_time_str = as_str_h(pred_time)
    #         direction, speed = compute(row[x], row[y])
    #         print(row['Station_Id'], row['XLONG'], row['XLAT'], row['SpotTime'], pred_time_str, direction, speed)
    #         after_df = after_df.append([{'Station_Id': row.Station_Id,
    #                                      'XLONG': row.XLONG,
    #                                      'XLAT': row.XLAT,
    #                                      'SpotTime': row.SpotTime,
    #                                      'PredTime': pred_time_str,
    #                                      'Direction': direction,
    #                                      'Speed': speed,
    #                                      'UU': row[hour + 'h_6'],
    #                                      'VV': row[hour + 'h_7'],
    #                                      'Q2': row[hour + 'h_8'],
    #                                      'T2': row[hour + 'h_9'],
    #                                      'PSFC': row[hour + 'h_10'],
    #                                      'QVAPOR': row[hour + 'h_11'],
    #                                      'QCLOUD': row[hour + 'h_12'],
    #                                      'HGT': row[hour + 'h_13'],
    #                                      'RAINC': row[hour + 'h_14'],
    #                                      'RAINNC': row[hour + 'h_15'],
    #                                      'SWDOWN': row[hour + 'h_16'],
    #                                      'GSW': row[hour + 'h_17'],
    #                                      'GLW': row[hour + 'h_18'],
    #                                      'HFX': row[hour + 'h_19'],
    #                                      'QFX': row[hour + 'h_20'],
    #                                      'LH': row[hour + 'h_21'],
    #                                      'TT': row[hour + 'h_22'],
    #                                      'GHT': row[hour + 'h_23'],
    #                                      'RH': row[hour + 'h_24'],
    #                                      'SLP': row[hour + 'h_25']
    #                                      }], ignore_index=True)
    # # wrf output per hour
    # elif len(row) == 2931:
    #     for x, y in zip(u10_columns, v10_columns):
    #         if int(x.split('h')[0]) % 6 == 0:
    #             hour = x.split('h')[0]
    #             predTime = ymd_h(row['SpotTime']) + hours(int(hour))
    #             predTimeStr = as_str_h(predTime)
    #             direction, speed = compute(row[x], row[y])
    #             print(row['Station_Id'], row['XLONG'], row['XLAT'], row['SpotTime'], predTimeStr, direction, speed)
    #             after_df = after_df.append([{'Station_Id': row.Station_Id,
    #                                          'XLONG': row.XLONG,
    #                                          'XLAT': row.XLAT,
    #                                          'SpotTime': row.SpotTime,
    #                                          'PredTime': predTimeStr,
    #                                          'Direction': direction,
    #                                          'Speed': speed
    #                                          }], ignore_index=True)
