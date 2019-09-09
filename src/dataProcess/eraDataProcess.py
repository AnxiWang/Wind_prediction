import pickle
import os
import netCDF4 as nc
import csv
import time
import pandas as pd
import numpy as np


from scipy.interpolate import griddata

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
    # file_name = era_dir.split('/')[-1].split('.')[0]
    # out_name = '../../data/era_pre/' + file_name + '.csv'
    # era_df = pd.DataFrame(columns=['time', 'long', 'lat', 'u10', 'v10', 'd2m', 't2m', 'msl'])
    # time_set = time.localtime(float(time_set))
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


def flatten(csv):
    u10 = pd.read_csv(csv, encoding='utf-8')
    u10 = u10.drop(['Unnamed: 0'], axis=1)
    date_time = u10['time'].values
    print(date_time)


if __name__ == "__main__":
    # for year in range(2013, 2020, 1):
    #     read_ERA_to_csv(era_dir + str(year) + '.nc', str(year))
    flatten('../../data/output/u10.csv')
    # conf = configparser.ConfigParser()
    # conf.read(os.getcwd() + '/config/config.ini', encoding="utf-8")
    # analysis_dir = conf.get('data_dirs', 'analysis_dir')
    # ear_grid_on_wrf = conf.get('data_dirs', 'ear_grid_on_wrf')
    # wrf_dir = conf.get('data_dirs', 'wrf_dir')
    # data_from_analysis_to_wrf(analysis_dir, ear_grid_on_wrf, wrf_dir)
    # name = []
    # print(type(name))
    # read_ERA_to_csv(era_dir)
