import numpy as np
import pandas as pd
import tqdm
import pickle
import os

from dataProcessUtils import *

gts_dir = '../../data/wind/'
wrf_pre_dir = '../../data/wrfout_pre'


# load GTS data and gather into one file per year
def gatherGTS(year):
    print('***************Gather GTS data by year***************')
    outputFile = '../../data/GTS/GTS_' + str(year) + '_wind.csv'
    if os.path.isfile(outputFile):
        print('file exists')
        GTS = pd.read_csv(outputFile, encoding='utf-8')
        return GTS
    else:
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
        if not os.path.exists(outputFile):
            GTS.to_csv(outputFile, index=False)
        return GTS


# load GTS data and gather into one file per month
def gatherMonthGTS(month, year):
    print('***************Gather GTS data by month***************')
    outputFile = '../../data/GTS/GTS_' + str(year) + month + '_wind.csv'
    if os.path.isfile(outputFile):
        print('file exists')
        GTS = pd.read_csv(outputFile, encoding='utf-8')
        return GTS
    else:
        GTS = pd.DataFrame(columns=['stationID', 'LONG', 'LAT', 'Date', 'Hour', 'Direction',
                                    'Speed', 'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Time'])
        month_gts_dir = gts_dir + str(year) + '/' + str(year) + month
        day_gts_name = os.listdir(month_gts_dir)
        gts_times = get_gts_time(month_gts_dir).sort_values()
        print(gts_times)
        dayGTS = load_gts(month_gts_dir, gts_times)
        GTS = pd.concat([GTS, dayGTS])
        GTS = GTS.drop_duplicates(keep='first')
        GTS = remove_duplicate_gts(GTS)
        GTS = remove_abnormal(GTS)
        if not os.path.exists(outputFile):
            GTS.to_csv(outputFile, index=False)
        return GTS


# dump WRF data or read the dumped data (2013-2017)
def dumpWRF(wrf_dir, wrf_times, out_name, year):
    print('***************Dump or read WRF data***************')
    wrf_pred = []
    wrf_pred_full_path = wrf_pre_dir + '/' + out_name + '_pred_' + str(year) + '_6h.pkl'
    if os.path.isfile(wrf_pred_full_path):
        print("Loading wind forecast data...")
        with open(wrf_pred_full_path, 'rb') as of:
            wrf_pred = pickle.load(of)
    else:
        print("Extracting wind forecast data...")
        if out_name == 'wrf':
            wrf_pred = [dump_wrf_var(wrf_dir, wrf_times, wrf_times + hours(7 * 24 + 12), str(year))]
            with open(wrf_pred_full_path, 'wb') as of:
                pickle.dump(wrf_pred, of)
        elif out_name == 'pcwrf':
            wrf_pred = [dump_pcwrf_var(wrf_dir, wrf_times, wrf_times + hours(7 * 24 + 12), str(year))]
            with open(wrf_pred_full_path, 'wb') as of:
                pickle.dump(wrf_pred, of)
    return wrf_pred


# dump WRF data or read the dumped data (2018-2019)
def dumpMonthWRF(wrf_month_dir, wrf_month_times, out_name, month, year):
    print('***************Dump or read WRF data***************')
    wrf_pred = []
    wrf_pred_full_path = wrf_pre_dir + '/' + out_name + '_pred_' + str(year) + month + '_1h.pkl'
    if os.path.isfile(wrf_pred_full_path):
        print("Loading wind forecast data...")
        with open(wrf_pred_full_path, 'rb') as of:
            wrf_pred = pickle.load(of)
    else:
        print("Extracting wind forecast data...")
        if out_name == 'wrf':
            wrf_pred = [dump_wrf_var_month(wrf_month_dir, wrf_month_times, wrf_month_times + hours(7 * 24 + 12), month, str(year))]
            with open(wrf_pred_full_path, 'wb') as of:
                pickle.dump(wrf_pred, of)
        elif out_name == 'pcwrf':
            wrf_pred = [dump_pcwrf_var_month(wrf_month_dir, wrf_month_times, wrf_month_times + hours(7 * 24 + 12), month, str(year))]
            with open(wrf_pred_full_path, 'wb') as of:
                pickle.dump(wrf_pred, of)
    return wrf_pred


# interpolate year WRF data to the station and compute wind direction, speed from U,V
def meshWRFtoStation(wrf_pred, station, mesh, year):
    print('***************Handle WRF data***************')
    sta_wrf_pred_path = '../../data/output/sta_wrf_pred_' + str(year) + '.csv'
    sta_wrf_pred_from_UV_path = '../../data/output/sta_wrf_pred_DS_' + str(year) + '.csv'
    if os.path.isfile(sta_wrf_pred_path) and os.path.isfile(sta_wrf_pred_from_UV_path):
        print('file exists')
        sta_wrf_pred_from_UV = pd.read_csv(sta_wrf_pred_from_UV_path, encoding='utf-8')
        return sta_wrf_pred_from_UV
    else:
        sta_wrf_pred = wrf_mesh_to_station(station, wrf_pred, mesh)
        sta_wrf_pred.to_csv(sta_wrf_pred_path, index=False)
        sta_wrf_pred_from_UV = calculateSpeedFromUV(sta_wrf_pred)
        sta_wrf_pred_from_UV.to_csv(sta_wrf_pred_from_UV_path, encoding='utf-8')
        return sta_wrf_pred_from_UV


# interpolate month WRF data to the station and compute wind direction, speed from U,V
def meshMonthWRFtoStation(wrf_pred, station, mesh, month, year):
    print('***************Mesh WRF data***************')
    month_sta_wrf_pred_path = '../../data/output/sta_wrf_pred_' + str(year) + month + '.csv'
    month_sta_wrf_pred_from_UV_path = '../../data/output/sta_wrf_pred_DS_' + str(year) + month + '.csv'
    if os.path.isfile(month_sta_wrf_pred_path) and os.path.isfile(month_sta_wrf_pred_from_UV_path):
        print('file exists')
        month_sta_wrf_pred_from_UV = pd.read_csv(month_sta_wrf_pred_from_UV_path, encoding='utf-8')
        return month_sta_wrf_pred_from_UV
    else:
        month_sta_wrf_pred = wrf_mesh_to_station(station, wrf_pred, mesh)
        month_sta_wrf_pred.to_csv(month_sta_wrf_pred_path, index=False)
        month_sta_wrf_pred_from_UV = calculateSpeedFromUV(month_sta_wrf_pred)
        month_sta_wrf_pred_from_UV.to_csv(month_sta_wrf_pred_from_UV_path, encoding='utf-8')
        return month_sta_wrf_pred_from_UV


def buildDataset(year):
    print('***************Build year dataset***************')
    dataset_path = '../../data/output/dataset_' + str(year) + '.csv'
    if os.path.isfile(dataset_path):
        print('file exists')
        dataset = pd.read_csv(dataset_path, encoding='utf-8')
        return dataset
    else:
        WRF = pd.read_csv('../../data/output/sta_wrf_pred_DS_' + str(year) + '.csv', encoding='utf-8')
        GTS = pd.read_csv('../../data/GTS/GTS_' + str(year) + '_wind.csv', encoding='utf-8')
        res = WRF.merge(GTS, left_on=['Station_Id', 'PredTime'], right_on=['stationID', 'Time'])
        dataset = pd.DataFrame(res,
                               columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime',
                                        'Direction_x', 'Speed_x',
                                        'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Direction_y', 'Speed_y'])
        # fill nan
        dataset.SeaPressure.fillna(dataset.SeaPressure.mean(), inplace=True)
        dataset.StaPressure.fillna(dataset.StaPressure.mean(), inplace=True)
        dataset.P3.fillna(dataset.P3.mean(), inplace=True)
        dataset.Temp.fillna(dataset.Temp.mean(), inplace=True)
        dataset.DPT.fillna(dataset.DPT.mean(), inplace=True)
        dataset.fillna(0, inplace=True)

        # drop outlier data
        for index, var in dataset.iterrows():
            if var['Direction_y'] == 0 and var['Speed_y'] == 0:
                dataset.drop(index, inplace=True)
        dataset.to_csv(dataset_path, encoding='utf-8')
        return dataset


def buildMonthDataset(month, year):
    print('***************Build month dataset***************')
    dataset_path = '../../data/output/dataset_' + str(year) + month + '.csv'
    if os.path.isfile(dataset_path):
        print('file exists')
        dataset = pd.read_csv(dataset_path, encoding='utf-8')
        return dataset
    else:
        WRF = pd.read_csv('../../data/output/sta_wrf_pred_DS_' + str(year) + month + '.csv', encoding='utf-8')
        GTS = pd.read_csv('../../data/GTS/GTS_' + str(year) + month + '_wind.csv', encoding='utf-8')
        res = WRF.merge(GTS, left_on=['Station_Id', 'PredTime'], right_on=['stationID', 'Time'])
        dataset = pd.DataFrame(res,
                               columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime',
                                        'Direction_x', 'Speed_x',
                                        'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Direction_y', 'Speed_y'])
        # fill nan
        dataset.SeaPressure.fillna(dataset.SeaPressure.mean(), inplace=True)
        dataset.StaPressure.fillna(dataset.StaPressure.mean(), inplace=True)
        dataset.P3.fillna(dataset.P3.mean(), inplace=True)
        dataset.Temp.fillna(dataset.Temp.mean(), inplace=True)
        dataset.DPT.fillna(dataset.DPT.mean(), inplace=True)
        dataset.fillna(0, inplace=True)

        # drop outlier data
        for index, var in dataset.iterrows():
            if var['Direction_y'] == 0 and var['Speed_y'] == 0:
                dataset.drop(index, inplace=True)
        dataset.to_csv(dataset_path, encoding='utf-8')
        return dataset
