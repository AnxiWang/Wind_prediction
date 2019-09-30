# import pandas as pd
#
# sta_df = pd.read_csv('../../data/station/cityStation.csv')
# fun = sta_df['stationID'].apply(lambda x: x.isdigit())
# grid = sta_df[fun == True]
# print(grid)

# from dataProcessUtils import ymd, hours, get_wrf_time, as_str_h, as_str
# import datetime
# start = '2019-04-07'
# end = '2019-04-14'
#
# date_start = datetime.datetime.strptime(start, '%Y-%m-%d')
# date_end = datetime.datetime.strptime(end, '%Y-%m-%d')
#
# while date_start < date_end:
#     date_start += datetime.timedelta(days=1)
#     print(as_str(date_start))

# import pandas as pd
# GTS = pd.read_csv('../../data/gts.csv', encoding='utf-8')
# WRF = pd.read_csv('../../data/wrf_predict.csv', encoding='utf-8')
# # 6个小时输出一个
# for index, var in WRF.iterrows():
#     if int(str(int(var['PredTime']))[8:10]) % 6 != 0:
#         print(var['PredTime'])

# import os
# import pickle
# import numpy as np
# import pandas as pd
#
#
# def find_nearest_point(p, grid):
#     grid['dist'] = (grid['lon'] - p[0]) ** 2 + (grid['lat'] - p[1]) ** 2
#     return grid.nsmallest(4, columns='dist')
#
#
# u10_col = []
# v10_col = []
# for col_i in range(0, 21, 1):
#     u10_col.append(str(col_i * 6) + 'h')
#     v10_col.append(str(col_i * 6) + 'h_v10')
# u10_col_1h = []
# v10_col_1h = []
# for col_i in range(0, 133, 1):
#     u10_col_1h.append(str(col_i) + 'h')
#     v10_col_1h.append(str(col_i) + 'h_v10')
# wrf_pred = []
# wrf_pred_full_path = '../../data/wrfout_pre/wrf_pred_2013.pkl'
# station = pd.read_csv('../../data/station/sta_43.csv', encoding='utf-8')
# mesh = pd.read_csv('../../data/output/mesh.csv', encoding='utf-8')
# if os.path.isfile(wrf_pred_full_path):
#     print("Loading wind forecast data...")
#     with open(wrf_pred_full_path, 'rb') as of:
#         wrf_pred = pickle.load(of)
# res = []
# ncNumber = len(wrf_pred[0])
# nc_num = int(ncNumber / 172)  # 502
#
# for index in range(0, station.shape[0], 1):
#     s_lon = station.loc[index].LONG
#     s_lat = station.loc[index].LAT
#
#     mesh_00 = [wrf_pred[0][i] for i in range((4 * index + 0) * nc_num, (4 * index + 1) * nc_num, 1)]
#     mesh_01 = [wrf_pred[0][i] for i in range((4 * index + 1) * nc_num, (4 * index + 2) * nc_num, 1)]
#     mesh_02 = [wrf_pred[0][i] for i in range((4 * index + 2) * nc_num, (4 * index + 3) * nc_num, 1)]
#     mesh_03 = [wrf_pred[0][i] for i in range((4 * index + 3) * nc_num, (4 * index + 4) * nc_num, 1)]
#
#     nn = find_nearest_point([s_lon, s_lat], mesh)
#     nn['weight'] = (1 / nn["dist"]) / sum(1 / nn["dist"])
#     weight = np.array(nn['weight'])
#     each_res = []
#
#     for each_nc in range(0, nc_num, 1):
#         t1_0 = mesh_00[each_nc][0]
#         t2_0 = mesh_00[each_nc][1]
#         df = pd.DataFrame(
#             {'Station_Id': station.loc[index].stationID, 'XLONG': s_lon, 'XLAT': s_lat, 'SpotTime': t1_0,
#              'PredEndTime': t2_0}, index=[0])
#         for var in range(4, 26, 1):
#             var_0 = mesh_00[each_nc][var]
#             var_1 = mesh_01[each_nc][var]
#             var_2 = mesh_02[each_nc][var]
#             var_3 = mesh_03[each_nc][var]
#
#             key = var_0 * weight[0] + var_1 * weight[1] + var_2 * weight[2] + var_3 * weight[3]
#             print(key)
#
#             if len(key) == 21:
#                 key_pd = pd.DataFrame([key], columns=[x + '_' + str(var) for x in u10_col])
#                 df = df.join(key_pd, how='right')
#             elif len(key) < 21:
#                 key = key.tolist()
#                 key.extend(np.nan for _ in range(21 - len(key)))
#                 key_pd = pd.DataFrame([key], columns=[x + '_' + str(var) for x in u10_col])
#                 df = df.join(key_pd, how='right')
#             elif 21 < len(key) <= 133:
#                 key = key.tolist()
#                 key.extend(np.nan for _ in range(133 - len(key)))
#                 key_pd_1h = pd.DataFrame([key], columns=[x + '_' + str(var) for x in u10_col_1h])
#                 df = df.join(key_pd_1h, how='right')
#             else:
#                 print('error!')
#         each_res.append(df)
#     res.append(each_res)
# result = pd.concat(res, sort=False).reset_index().drop(columns=['index'])
# result.to_csv('mesh_sta.csv', encoding='utf-8')


# import pandas as pd
#
# test = pd.read_csv('../../data/output/sta_wrf_pred_2013.csv', encoding='utf-8')
# for index, var in test.iterrows():
#     print(var)

# import os
# if os.path.isfile('../../data/wind/GTS.out_201305_wind.csv'):
#     print('exist')
# else:
#     print('not exist')

# -*- coding:utf-8 -*-

# import time
# import multiprocessing

# def job(x, y):
#     return y + x
#
#
# def job1(z):
#     return job(z[0], z[1])
#
#
# if __name__ == "__main__":
#     dataSetPath = []
#     year = []
#     for i in range(2013, 2020, 1):
#         dataSetPath.append(str(i))
#         year.append(str(i))
#
#     param = zip(dataSetPath, year)
#     with multiprocessing.Pool(10) as p:
#         res = p.map(job1, param)

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
#
#
# def func(x, y):
#     return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2
#
#
# grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
# points = np.random.rand(1000, 2)
# values = func(points[:, 0], points[:, 1])
#
# grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
# grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
# grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
#
# plt.subplot(221)
# plt.imshow(func(grid_x, grid_y).T, extent=(0, 1, 0, 1), origin='lower')
# plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
#
# plt.title('Original')
# plt.subplot(222)
# plt.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
#
# plt.title('Nearest')
# plt.subplot(223)
# plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')
#
# plt.title('Linear')
# plt.subplot(224)
# plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
#
# plt.title('Cubic')
# plt.gcf().set_size_inches(6, 6)
# plt.show()

# import pandas as pd
# # res = []
# # list1 = [[1, 2, 3, 4]]
# # list1_pd = pd.DataFrame(list1)
# # res.append(list1_pd)
# # print(pd.concat(res, sort=False).reset_index().drop(columns=['index']))
# import pandas as pd
# mesh = pd.read_csv('../../data/output/near_mesh.csv')
# dataset = mesh[(mesh['sta_lon'] == 121.7) & (mesh['sta_lat'] == 39.08)]
# print(dataset)

import pandas as pd
import os

path = '../../data/predict'
files = os.listdir(path)

df1 = pd.read_csv(path + '/' + files[0], encoding='utf-8', index_col=0)

for file in files[1:]:
    df2 = pd.read_csv(path + '/' + file, encoding='utf-8', index_col=0)
    df1 = pd.concat([df1, df2], axis=0, ignore_index=True)

df1 = df1.drop_duplicates()
df1 = df1.reset_index(drop=True)
df1.to_csv('../../data/total.csv')

# outputfile = '../../data/pred_res.csv'
# for inputfile in os.listdir('../../data/predict'):
#     df = pd.read_csv('../../data/predict/' + inputfile, header=None)
#     df.to_csv(outputfile, mode='a+', index=False, header=False)
