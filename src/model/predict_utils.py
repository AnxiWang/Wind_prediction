import os
import sys
sys.path.append('../dataProcess/')
import pandas as pd
from netCDF4 import Dataset
import glob
import datetime as dt
import os.path
import pickle
import tqdm
import datetime
import numpy as np
from dataProcessUtils import ymd, hours, get_wrf_time, as_str_h, as_str, build_mesh, find_nearest_point, calculateSpeedFromUV
from windFeature import getWindInfo
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
gts_pre_dir = '../../data/gts_pre'


def wrf_mesh_to_station(station, wrf_pred, mesh):
    sta_wind_pred = []
    for index in range(0, station.shape[0], 1):
        s_lon = station.loc[index].LONG
        s_lat = station.loc[index].LAT
        # 太平洋和印度洋数据重叠，如何划分对应的站点？
        nn = find_nearest_point([s_lon, s_lat], mesh)
        nn['weight'] = (1 / nn["dist"]) / sum(1 / nn["dist"])
        ilon = np.array(nn['ilon'], dtype=int)
        ilat = np.array(nn['ilat'], dtype=int)
        weight = np.array(nn['weight'])
        t1 = np.array([x[0] for x in wrf_pred])
        t2 = np.array([x[1] for x in wrf_pred])
        u0 = np.array([x[4][:, ilat[0], ilon[0]] for x in wrf_pred])
        u1 = np.array([x[4][:, ilat[1], ilon[1]] for x in wrf_pred])
        u2 = np.array([x[4][:, ilat[2], ilon[2]] for x in wrf_pred])
        u3 = np.array([x[4][:, ilat[3], ilon[3]] for x in wrf_pred])
        # print(u0, u1, u2, u3)

        v0 = np.array([x[5][:, ilat[0], ilon[0]] for x in wrf_pred])
        v1 = np.array([x[5][:, ilat[1], ilon[1]] for x in wrf_pred])
        v2 = np.array([x[5][:, ilat[2], ilon[2]] for x in wrf_pred])
        v3 = np.array([x[5][:, ilat[3], ilon[3]] for x in wrf_pred])
        # print(v0, v1, v2, v3)

        u = u0 * weight[0] + u1 * weight[1] + u2 * weight[2] + u3 * weight[3]
        v = v0 * weight[0] + v1 * weight[1] + v2 * weight[2] + v3 * weight[3]

        df = pd.DataFrame(
            {'Station_Id': station.loc[index].stationID, 'XLONG': s_lon, 'XLAT': s_lat, 'SpotTime': t1,
             'PredEndTime': t2})
        if len(u[0]) == 21:
            u10_pd = pd.DataFrame(u, columns=u10_col)
            v10_pd = pd.DataFrame(v, columns=v10_col)
            df = df.join(u10_pd, how='right')
            df = df.join(v10_pd, how='right')
            sta_wind_pred.append(df)
        elif len(u[0]) < 21:
            u = u.tolist()
            v = v.tolist()
            u[0].extend(np.nan for _ in range(21 - len(u[0])))
            v[0].extend(np.nan for _ in range(21 - len(v[0])))

            u10_pd = pd.DataFrame(u, columns=u10_col)
            v10_pd = pd.DataFrame(v, columns=v10_col)
            df = df.join(u10_pd, how='right')
            df = df.join(v10_pd, how='right')
            sta_wind_pred.append(df)
        elif 21 < len(u[0]) <= 133:
            u = u.tolist()
            v = v.tolist()
            u[0].extend(np.nan for _ in range(133 - len(u[0])))
            v[0].extend(np.nan for _ in range(133 - len(v[0])))

            u10_pd_1h = pd.DataFrame(u, columns=u10_col_1h)
            v10_pd_1h = pd.DataFrame(v, columns=v10_col_1h)
            df = df.join(u10_pd_1h, how='right')
            df = df.join(v10_pd_1h, how='right')
            sta_wind_pred.append(df)
        else:
            print('error!')

    return pd.concat(sta_wind_pred).reset_index().drop(columns=['index'])


def latest_wrf_date(wrf_dir):
    """
    :param wrf_dir: WRF数据存放路径，只获取到文件夹的名称，即日期
    :return: 最新的WRF数据日期
    """
    wrf_times = get_wrf_time(wrf_dir).sort_values()
    return pd.to_datetime(np.sort(wrf_times)[-1])


def get_wrf_path(start, end):
    return 'pwrfout_d01.{0}_{1}.nc'.format(as_str_h(start), as_str_h(end))


# load model
def load_model():
    # 根据预测时长读取对应的模型
    files = glob.glob("../../data/model/*.pkl")
    dates = [i[23:31] for i in files]
    latest = max(dates)
    model_filename = '../../data/model/model-{0}_lead_7.pkl'.format(latest)
    print(model_filename)
    with open(model_filename, 'rb') as of:
        return pickle.load(of)


def ymd_h(x):
    return pd.to_datetime(x, format="%Y%m%d%H")


# dump wrf data into one file
def get_wrf_var(wrf_file_dir, start, end):
    res = []
    with Dataset(wrf_file_dir, mode='r', format='NETCDF4_CLASSIC') as ds:
        variable_keys = ds.variables.keys()
        if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys and 'XLAT' in variable_keys:
            res.append([ymd_h(start).strftime(ymdh),
                        (ymd_h(end) - hours(2 * 24)).strftime(ymdh),
                        ds['XLONG'][:].data,
                        ds['XLAT'][:].data,
                        ds['U10'][:].data,
                        ds['V10'][:].data])
    return res


def load_gts(gts_pre_dir, gts_times):
    gts_files = [gts_pre_dir + '/' + x[0:4] + '/' + x[0:6] + '/GTS.out_' + x + '_wind.csv' for x in gts_times]
    res = [pd.read_csv(f, encoding='windows-1252') for f in gts_files]
    total = pd.concat(res)
    for c in total.columns:
        if total[c].dtype.name == 'float64':
            total[c] = total[c].astype('float32')
        if total[c].dtype.name == 'int64':
            total[c] = total[c].astype('int32')
    total.reset_index(inplace=True)
    # total['Time'] = pd.to_datetime(total['Time'])
    total['stationID'] = total['stationID'].astype('str')
    return total


def load_latest_gts(base_gts_dir, gts_times):
    for i in range(len(gts_times)):
        out_dir = gts_pre_dir + '/' + 'GTS.out_' + gts_times[i] + '_wind.csv'
        gts_dir = base_gts_dir + '/' + gts_times[i][0:4] + '/' + gts_times[i][0:6] + '/' + 'GTS.out_' + gts_times[i]
        getWindInfo(gts_dir, out_dir, gts_times[i][0:4])
