import numpy as np
import pandas as pd
import tqdm
import pickle
import os

from dataProcessUtils import *

gts_dir = '../../data/wind/2013'
wrf_dir = '../../data/wrfout_6h/'
wrf_pre_dir = '../../data/wrfout_pre'
station_dir = '../../data/station/IndianStation_test.csv'

# 生成网格
# mesh = build_mesh()
# mesh.to_csv('../../data/output/mesh.csv')
# wrf_times = get_wrf_time(wrf_dir).sort_values()
# print(wrf_times)
# 加载GTS数据，一年合并为一个文件
# GTS = pd.DataFrame(columns=['stationID', 'LONG', 'LAT', 'Date', 'Hour', 'Direction', 'Speed', 'SeaPressure', 'StaPressure', 'P3',
#                  'Temp', 'DPT', 'Time'])
# month_gts_name = os.listdir(gts_dir)
# for month in month_gts_name:
#     gts_times = get_gts_time(gts_dir + '/' + month).sort_values()
#     print(gts_times)
    # monthGTS = load_gts(gts_dir + '/' + month, gts_times)
    # GTS = pd.concat([GTS, monthGTS])
# print(GTS)
# GTS = GTS.drop_duplicates(keep='first')
# GTS.to_csv('../../data/GTS/GTS_2013_wind.csv', index=False)
# gts_times = gts_times[gts_times > ymd('20130101')]


# 加载GTS数据
# print(GTS)
# GTS.to_csv('../../data/GTS/GTS_wind.csv', index=False)

# GTS = remove_duplicate_gts(GTS)
# GTS.to_csv('../../data/GTS/GTS_wind_remove.csv', index=False)
# print("Cleaning abnormal values...")
# GTS = remove_abnormal(GTS)
# GTS.to_csv('../../data/GTS/GTS_wind_normal.csv', index=False)

# WRF模式数据读取
# wrf_pred = []
# wrf_pred_full_path = wrf_pre_dir + '/wrf_pred_2013_6h.pkl'
# if os.path.isfile(wrf_pred_full_path):
#     print("Loading wind forecast data...")
#     with open(wrf_pred_full_path, 'rb') as of:
#         wrf_pred = pickle.load(of)
# else:
#     print("Extracting wind forecast data...")
#     wrf_pred = [dump_wrf_var(wrf_dir, wrf_times, wrf_times + hours(7 * 24 + 12))]
#     with open(wrf_pred_full_path, 'wb') as of:
#         pickle.dump(wrf_pred, of)


# 观测数据缺失值处理
# GTS = GTS[GTS.Time >= ymd("20170101")]
# print("Fixing NA values in wind data...")
# GTS.Precipitation_1H.fillna(0, inplace=True)
# GTS.Precipitation_3H.fillna(GTS.Precipitation_1H, inplace=True)
# GTS.Precipitation_6H.fillna(GTS.Precipitation_3H, inplace=True)
# GTS.Precipitation_12H.fillna(GTS.Precipitation_6H, inplace=True)
# GTS.Precipitation_24H.fillna(GTS.Precipitation_12H, inplace=True)
# 读取站点信息
# print("Getting station infomation from station data...")
# station = pd.read_csv(station_dir, encoding='windows-1252')
# mesh = pd.read_csv('../../data/output/mesh.csv', encoding='utf-8')
# GTS = pd.read_csv('../../data/GTS/GTS_wind_normal.csv', encoding='utf-8')
# sta_wrf_pred = pd.read_csv('../../data/output/sta_wrf_pred.csv', encoding='utf-8')
#
# station = station[['stationID', 'LONG', 'LAT']] \
#     .groupby("stationID") \
#     .head(1) \
#     .reset_index() \
#     .drop(columns='index')
# print(station)
# WRF数据中的U,V转化为风速和风向
# sta_wrf_pred = wrf_mesh_to_station(station, wrf_pred, mesh)
# sta_wrf_pred.to_csv('../../data/output/sta_wrf_pred.csv', index=False)
# sta_wrf_pred_from_UV = calculateSpeedFromUV(sta_wrf_pred)
# sta_wrf_pred_from_UV.to_csv('../../data/output/sta_wrf_pred_DS.csv', encoding='utf-8')
# 合并GTS和WRF数据
GTS = pd.read_csv('../../data/GTS/GTS_2013_wind.csv', encoding='utf-8')
WRF = pd.read_csv('../../data/output/sta_wrf_pred_DS.csv', encoding='utf-8')
res = WRF.merge(GTS, left_on=['Station_Id', 'PredTime'], right_on=['stationID', 'Time'])
res.to_csv('../../data/output/dataset_all.csv', encoding='utf-8')

inner_df = pd.read_csv('../../data/output/dataset_all.csv', encoding='utf-8', dtype={'SpotTime': int})
dataset = pd.DataFrame(inner_df,
                       columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime',
                                'Direction_x', 'Speed_x',
                                'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Direction_y', 'Speed_y'])
# 缺失值处理
dataset.fillna(0, inplace=True)
# dataset.StaPressure.fillna(0, inplace=True)
# dataset.P3.fillna(0, inplace=True)

dataset.to_csv('../../data/output/dataset.csv', encoding='utf-8')
