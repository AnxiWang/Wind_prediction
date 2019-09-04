import numpy as np
import pandas as pd
import tqdm
import pickle
import os

from dataProcessUtils import *

gts_dir = '../../data/wind/'
wrf_dir = '/home/shared_data/external/IDWRF/202.108.199.14/IDWRF/OUTPUT_P/PACK_IDWRF_6h/'
wrf_pre_dir = '../../data/wrfout_pre'
station_dir = '../../data/station/IndianStation_test.csv'

# 生成网格
mesh = build_mesh()
mesh.to_csv('../../data/output/mesh.csv')

# 读取站点信息
station = pd.read_csv(station_dir, encoding='windows-1252')
station = station[['stationID', 'LONG', 'LAT']] \
    .groupby("stationID") \
    .head(1) \
    .reset_index() \
    .drop(columns='index')

wrf_times = get_wrf_time(wrf_dir).sort_values()
# print(wrf_times)


# 加载GTS数据，一年合并为一个文件
def gatherGTS(year):
    print('***************Gather GTS data***************')
    GTS = pd.DataFrame(columns=['stationID', 'LONG', 'LAT', 'Date', 'Hour', 'Direction',
                                'Speed', 'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Time'])
    year_gts_dir = gts_dir + str(year)
    month_gts_name = os.listdir(year_gts_dir)
    for month in month_gts_name:
        gts_times = get_gts_time(year_gts_dir + '/' + month).sort_values()
        print(gts_times)
        monthGTS = load_gts(year_gts_dir + '/' + month, gts_times)
        GTS = pd.concat([GTS, monthGTS])
    GTS = GTS.drop_duplicates(keep='first')
    GTS = remove_duplicate_gts(GTS)
    GTS = remove_abnormal(GTS)
    # print(GTS)
    outputFile = '../../data/GTS/GTS_' + str(year) + '_wind.csv'
    if not os.path.exists(outputFile):
        GTS.to_csv(outputFile, index=False)
    return GTS


# WRF模式数据读取
def dumpWRF(year):
    print('***************Dump or read WRF data***************')
    wrf_pred = []
    wrf_pred_full_path = wrf_pre_dir + '/wrf_pred_' + str(year) + '_6h.pkl'
    if os.path.isfile(wrf_pred_full_path):
        print("Loading wind forecast data...")
        with open(wrf_pred_full_path, 'rb') as of:
            wrf_pred = pickle.load(of)
    else:
        print("Extracting wind forecast data...")
        wrf_pred = [dump_wrf_var(wrf_dir, wrf_times, wrf_times + hours(7 * 24 + 12))]
        with open(wrf_pred_full_path, 'wb') as of:
            pickle.dump(wrf_pred, of)
    return wrf_pred


# wrf网格数据插值到站点，同时实现U、V到风向风速的转化
def meshWRFtoStation(wrf_pred, year):
    sta_wrf_pred = wrf_mesh_to_station(station, wrf_pred, mesh)
    sta_wrf_pred.to_csv('../../data/output/sta_wrf_pred_' + str(year) + '.csv', index=False)
    sta_wrf_pred_from_UV = calculateSpeedFromUV(sta_wrf_pred)
    sta_wrf_pred_from_UV.to_csv('../../data/output/sta_wrf_pred_DS_' + str(year) + '.csv', encoding='utf-8')
    return sta_wrf_pred_from_UV


def buildDataset(year):
    WRF = pd.read_csv('../../data/output/sta_wrf_pred_DS_' + str(year) + '.csv', encoding='utf-8')
    GTS = pd.read_csv('../../data/GTS/GTS_' + str(year) + '_wind.csv', encoding='utf-8')
    res = WRF.merge(GTS, left_on=['Station_Id', 'PredTime'], right_on=['stationID', 'Time'])
    dataset = pd.DataFrame(res,
                           columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime',
                                    'Direction_x', 'Speed_x',
                                    'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Direction_y', 'Speed_y'])
    # 缺失值处理
    dataset.SeaPressure.fillna(dataset.SeaPressure.mean(), inplace=True)
    dataset.StaPressure.fillna(dataset.StaPressure.mean(), inplace=True)
    dataset.P3.fillna(dataset.P3.mean(), inplace=True)
    dataset.Temp.fillna(dataset.Temp.mean(), inplace=True)
    dataset.DPT.fillna(dataset.DPT.mean(), inplace=True)
    dataset.fillna(0, inplace=True)

    # 删除观测数据中风向和风速同时为0的数据
    for index, var in dataset.iterrows():
        if var['Direction_y'] == 0 and var['Speed_y'] == 0:
            dataset.drop(index, inplace=True)
    dataset.to_csv('../../data/output/dataset_' + str(year) + '.csv', encoding='utf-8')
    return dataset




# # print("Getting station infomation from station data...")
# mesh = pd.read_csv('../../data/output/mesh.csv', encoding='utf-8')
# GTS = pd.read_csv('../../data/GTS/GTS_2013_wind.csv', encoding='utf-8')
#
#
#
# sta_wrf_pred = pd.read_csv('../../data/output/sta_wrf_pred_2013.csv', encoding='utf-8')
# # WRF数据中的U,V转化为风速和风向
# # 合并GTS和WRF数据
# # GTS = pd.read_csv('../../data/GTS/GTS_2013_wind.csv', encoding='utf-8')
# WRF = pd.read_csv('../../data/output/sta_wrf_pred_DS_2013.csv', encoding='utf-8')
