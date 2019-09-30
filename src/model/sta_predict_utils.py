import os
import sys

sys.path.append('../dataProcess/')
import pandas as pd
from netCDF4 import Dataset
import glob
import datetime as dt
import os.path
import pickle
from tqdm import tqdm
import datetime
import numpy as np
from dataProcessUtils import ymd, hours, get_wrf_time, as_str_h, as_str, build_mesh, find_nearest_point, \
    calculateSpeedFromUV
from windFeature import getWindInfo

ymdh = "%Y%m%d%H"
u10_col = []
for col_i in range(0, 21, 1):
    u10_col.append(str(col_i * 6) + 'h')
u10_col_1h = []
for col_i in range(0, 133, 1):
    u10_col_1h.append(str(col_i) + 'h')
gts_pre_dir = '../../data/gts_pre'


def wrf_mesh_to_station(station, wrf_pred, id_mesh):
    res = []
    for index in range(0, station.shape[0], 1):
        s_lon = station.loc[index].LONG
        s_lat = station.loc[index].LAT

        mesh_00 = wrf_pred[4 * index + 0]
        mesh_01 = wrf_pred[4 * index + 1]
        mesh_02 = wrf_pred[4 * index + 2]
        mesh_03 = wrf_pred[4 * index + 3]

        nn = find_nearest_point([s_lon, s_lat], id_mesh)
        nn['weight'] = (1 / nn["dist"]) / sum(1 / nn["dist"])
        # print(nn)
        weight = np.array(nn['weight'])

        t1_0 = mesh_00[0]
        t2_0 = mesh_00[1]
        df = pd.DataFrame(
                {'Station_Id': station.loc[index].stationID, 'XLONG': s_lon, 'XLAT': s_lat, 'SpotTime': t1_0,
                 'PredEndTime': t2_0}, index=[0])
        for var in range(4, 26, 1):
            var_0 = mesh_00[var]
            var_1 = mesh_01[var]
            var_2 = mesh_02[var]
            var_3 = mesh_03[var]

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
    return pd.concat(res, sort=False).reset_index().drop(columns=['index'])


def latest_wrf_date(wrf_dir):
    """
    :param wrf_dir: WRF数据存放路径，只获取到文件夹的名称，即日期
    :return: 最新的WRF数据日期
    """
    wrf_times = get_wrf_time(wrf_dir).sort_values()
    return pd.to_datetime(np.sort(wrf_times)[-1])


def get_wrf_path(start, end, year):
    if year == '2019':
        return 'pwrfout_d01.{0}_{1}.nc'.format(as_str_h(start), as_str_h(end))
    elif year == '2018':
        return 'pack.pwrfout_d01.{0}_{1}.nc'.format(as_str_h(start), as_str_h(end))


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


def store_sta_wrf(sta_wrf_pred_from_UV, start, end, output_dir):
    file_name = 'wrf_{0}_{1}.csv'.format(as_str_h(start), as_str_h(end))
    out_file_path = output_dir + '/' + file_name
    if os.path.isfile(out_file_path):
        print('file exists')
    else:
        sta_wrf_pred_from_UV.to_csv(out_file_path, encoding='utf-8', index=0)


def get_sta_wrf_var(wrf_file_dir, near_mesh, start, end, year):
    res = []
    wrf_file_path = wrf_file_dir + '/' + get_wrf_path(start, end, year)
    print(wrf_file_path)
    ds = Dataset(wrf_file_path, mode='r', format='NETCDF4_CLASSIC')
    for index, var in tqdm(near_mesh.iterrows(), total=near_mesh.shape[0]):
        variable_keys = ds.variables.keys()
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
                res.append([ymd_h(start).strftime(ymdh),
                            (ymd_h(end) - hours(2 * 24)).strftime(ymdh),
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
    return res
