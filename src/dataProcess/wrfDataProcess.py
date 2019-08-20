import numpy as np
import pandas as pd
import tqdm
import pickle
import os

from dataProcessUtils import *

gts_dir = '../../data/wind'
wrf_dir = '../../data/wrfout/'
wrf_pre_dir = '../../data/wrfout_pre'
station_dir = '../../data/station/IndianStation.csv'

# 生成网格
# mesh = build_mesh()
# mesh.to_csv('../../data/output/mesh.csv')
# wrf_times = get_wrf_time(wrf_dir).sort_values()
# print(wrf_times)
# print(wrf_times)
# gts_times = get_gts_time(gts_dir).sort_values()
# print(gts_times)
# gts_times = gts_times[gts_times > ymd('20130101')]


# 加载GTS数据
# GTS = load_gts(gts_dir, gts_times)
# GTS.to_csv('../../data/GTS/GTS_wind.csv', index=False)

# GTS = remove_duplicate_gts(GTS)
# GTS.to_csv('../../data/GTS/GTS_wind_remove.csv', index=False)
# print("Cleaning abnormal values...")
# GTS = remove_abnormal(GTS)
# GTS.to_csv('../../data/GTS/GTS_wind_normal.csv', index=False)

# wrf模式数据读取
# wrf_pred = []
# wrf_pred_full_path = wrf_pre_dir + '/wrf_pred.pkl'
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

print("Getting station infomation from station data...")
station = pd.read_csv(station_dir, encoding='windows-1252')

station = station[['stationID', 'LONG', 'LAT']] \
    .groupby("stationID") \
    .head(1) \
    .reset_index() \
    .drop(columns='index')
print(station)

# sta_tp_pred = tp_mesh_to_station(GTS, tp_pred[lead - 1], mesh)
