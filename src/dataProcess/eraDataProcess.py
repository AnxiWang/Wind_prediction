import pickle
import os
import netCDF4 as nc
import csv
import time
import datetime as dt
import pandas as pd
import numpy as np
import re
from dataProcessUtils import hours, as_str_h, compute


from scipy.interpolate import griddata

t1 = dt.datetime(1900, 1, 1, 0, 0, 0)
# era_dir = '../../data/2017.nc'
era_dir = '/home/shared_data/era_4_moment/'
era_pre_dir = '../../data/era_pre'
mesh = pd.read_csv('../../data/output/mesh.csv', encoding='utf-8')

station = pd.read_csv('../../data/station/IndianStation_test.csv', encoding='utf-8')
s_lon = station.loc[:].LONG
s_lat = station.loc[:].LAT


def read_ERA_to_csv(era_dir, year):
    dataset = nc.Dataset(era_dir, 'r')
    # 获取相应数组集合
    lon_set = dataset['longitude'][:].data
    lat_set = dataset['latitude'][:].data
    time_set = dataset['time'][:].data
    u10_set = dataset['u10']
    v10_set = dataset['v10']
    t2m_set = dataset['t2m']
    msl_set = dataset['msl']

    date_df = pd.DataFrame(time_set, columns=['time'])

    lon = lon_set[115:526]
    lat = lat_set[230:446]
    d = np.meshgrid(lon, lat)

    res_u10 = []
    for index in range(u10_set.shape[0]):
        print("start to analyse u10 ,index is : {}".format(index))
        grid_data = griddata((d[0].ravel(), d[1].ravel()), u10_set[index, 230:446, 115:526].ravel(),
                             (s_lon, s_lat), method='nearest')
        res_u10.append(grid_data)
    u10_df = pd.DataFrame(res_u10, columns=station.loc[:].stationID)
    u10 = date_df.join(u10_df)    
    u10.to_csv('../../data/era_pre/u10_' + year + '.csv', index=False, encoding='utf-8')

    res_vl0 = []
    for index in range(v10_set.shape[0]):
        print("start to analyse v10 ,index is : {}".format(index))
        grid_data = griddata((d[0].ravel(), d[1].ravel()), v10_set[index, 230:446, 115:526].ravel(),
                             (s_lon, s_lat), method='nearest')
        res_vl0.append(grid_data)
    v10_df = pd.DataFrame(res_vl0, columns=station.loc[:].stationID)
    v10 = date_df.join(v10_df)
    v10.to_csv('../../data/era_pre/v10_' + year + '.csv', index=False, encoding='utf-8')

    res_t2m = []
    for index in range(t2m_set.shape[0]):
        print("start to analyse t2m ,index is : {}".format(index))
        grid_data = griddata((d[0].ravel(), d[1].ravel()), t2m_set[index, 230:446, 115:526].ravel(),
                             (s_lon, s_lat), method='nearest')
        res_t2m.append(grid_data)
    t2m_df = pd.DataFrame(res_t2m, columns=station.loc[:].stationID)
    t2m = date_df.join(t2m_df)
    t2m.to_csv('../../data/era_pre/t2m_' + year + '.csv', index=False, encoding='utf-8')

    res_msl = []
    for index in range(msl_set.shape[0]):
        print("start to analyse msl ,index is : {}".format(index))
        grid_data = griddata((d[0].ravel(), d[1].ravel()), msl_set[index, 230:446, 115:526].ravel(),
                             (s_lon, s_lat), method='nearest')
        res_msl.append(grid_data)
    msl_df = pd.DataFrame(res_msl, columns=station.loc[:].stationID)
    msl = date_df.join(msl_df)
    msl.to_csv('../../data/era_pre/msl_' + year + '.csv', index=False, encoding='utf-8')


def change_time(csv_path, file):
    date = []
    csv_target = pd.read_csv(csv_path, encoding='utf-8')
    # csv_target = csv_target.drop(['Unnamed: 0'], axis=1)
    date_time = csv_target['time'].values
    csv_target = csv_target.drop(['time'], axis=1)

    for i in range(len(date_time)):
        struct_time = as_str_h(t1 + hours(float(date_time[i])))
        date.append(struct_time)
    new_date = pd.DataFrame(date, columns=['time'])
    csv_target = new_date.join(csv_target)
    csv_target.to_csv('../../data/era_pre/out/new_' + file, index=False, encoding='utf-8')


def flatten(path, file):
    column_name = file.split('_')[1]
    stationID = []
    target_df = pd.read_csv(path, index_col=['time'], encoding='utf-8')
    row_count = target_df.shape[0]
    columns = list(target_df)
    for j in range(len(columns)):
        for i in range(row_count):
            stationID.append(columns[j])
    station_ID = pd.DataFrame(stationID, columns=['stationID'])
    # print(station_ID)
    df2 = pd.concat(target_df.iloc[:, i] for i in range(target_df.shape[1]))
    # print(df2)
    dict_time = {'time': df2.index, column_name: df2.values}
    df_time = pd.DataFrame(dict_time)
    new_df = station_ID.join(df_time)
    new_df.to_csv('../../data/era_pre/out_one/' + file[4:], index=False, encoding='utf-8')


def gather_era(path, files, year):
    year_df = pd.read_csv(path + '/u10_' + year + '.csv', encoding='utf-8')

    for file in files:
        if file.split('_')[1] == year + '.csv':
            file_df = pd.read_csv(path + '/' + file, encoding='utf-8')
            year_df = year_df.merge(file_df)
    year_df.to_csv('../../data/era_pre/' + year + '.csv', index=False, encoding='utf-8')


def compute_era_uv(year, year_file):
    after_df = pd.DataFrame(columns=['Station_Id', 'Time', 'Direction', 'Speed', 'MSL', 'T2M'])
    year_df = pd.read_csv(year_file, encoding='utf-8')
    for index, row in year_df.iterrows():
        direction, speed = compute(row['u10'], row['v10'])
        # print(direction, speed)
        after_df = after_df.append([dict(Station_Id=row.stationID, Time=row.time, Direction=direction, Speed=speed,
                                         MSL=row.msl, T2M=row.t2m)], ignore_index=True)
    after_df.to_csv('../../data/era_pre/era_' + year + '.csv', encoding='utf-8')


if __name__ == "__main__":
    # 从nc文件抽取需要的信息
    # for year in range(2013, 2020, 1):
    #     read_ERA_to_csv(era_dir + str(year) + '.nc', str(year))
    # 将抽取信息中的时间改为日期显示
    # files = os.listdir('../../data/era_pre/old')
    # for file in files:
    #     print(file)
    #     change_time('../../data/era_pre/old/' + file, file)
    # 将要素值展平
    # files = os.listdir('../../data/era_pre/out')
    # for file in files:
    #     # print(file)
    #     flatten('../../data/era_pre/out/' + file, file)
    # 合并
    # for year in range(2013, 2020, 1):
    #     path = '../../data/era_pre/out_one'
    #     files = os.listdir(path)
    #     gather_era(path, files, str(year))
    # 计算风速和风向
    for year in range(2013, 2020, 1):
        path = '../../data/era_pre/'
        year_file = path + str(year) + '.csv'
        compute_era_uv(str(year), year_file)
    # conf = configparser.ConfigParser()
    # conf.read(os.getcwd() + '/config/config.ini', encoding="utf-8")
    # analysis_dir = conf.get('data_dirs', 'analysis_dir')
    # ear_grid_on_wrf = conf.get('data_dirs', 'ear_grid_on_wrf')
    # wrf_dir = conf.get('data_dirs', 'wrf_dir')
    # data_from_analysis_to_wrf(analysis_dir, ear_grid_on_wrf, wrf_dir)
    # name = []
    # print(type(name))
