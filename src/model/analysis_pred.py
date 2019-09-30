import os
import sys

sys.path.append('../dataProcess/')
import pandas as pd
import sklearn.metrics
import numpy as np
import warnings

warnings.filterwarnings("ignore")
import configparser

conf = configparser.ConfigParser()
conf.read('model_config.ini', encoding="utf-8")

res_dir = conf.get('res_dirs', 'res_dir')
res_all_in_one = conf.get('res_dirs', 'res_all_in_one')
gts_dir = conf.get('res_dirs', 'gts_dir')
gts_all_in_one = conf.get('res_dirs', 'gts_all_in_one')
sta_dir = conf.get('data_dirs', 'station_dir')

features = ['Direction', 'Speed', 'UU', 'VV', 'Q2', 'T2', 'PSFC', 'QVAPOR', 'QCLOUD', 'HGT', 'RAINC', 'RAINNC',
            'SWDOWN', 'GSW', 'GLW', 'HFX', 'QFX', 'LH', 'TT', 'GHT', 'RH', 'SLP']
sta_list = pd.read_csv(sta_dir, encoding='utf-8', index_col=0)


def all_res_to_one():
    files = os.listdir(res_dir)

    df1 = pd.read_csv(res_dir + '/' + files[0], encoding='utf-8', index_col=0)

    for file in files[1:]:
        df2 = pd.read_csv(res_dir + '/' + file, encoding='utf-8', index_col=0)
        df1 = pd.concat([df1, df2], axis=0, ignore_index=True)

    df1 = df1.drop_duplicates()
    df1 = df1.reset_index(drop=True)
    df1.to_csv(res_all_in_one)


def all_gts_to_one():
    files = os.listdir(gts_dir)

    df1 = pd.read_csv(gts_dir + '/' + files[0], encoding='utf-8')

    for file in files[1:]:
        df2 = pd.read_csv(gts_dir + '/' + file, encoding='utf-8')
        df1 = pd.concat([df1, df2], axis=0, ignore_index=True)

    df1 = df1.drop_duplicates()
    df1 = df1.reset_index(drop=True)
    df1.to_csv(gts_all_in_one)


def analysis_predict():
    pred_sta = pd.read_csv(res_all_in_one, encoding='utf-8', index_col=0)
    gts_sta = pd.read_csv(gts_all_in_one, encoding='utf-8', index_col=0)
    pred_sta = pred_sta[['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime',
                         'Direction', 'Speed', 'T2', 'SLP',
                         'seapre_predict', 'temp_predict', 'dir_predict', 'speed_predict']]
    gts_sta = gts_sta[['stationID', 'Time', 'Direction', 'Speed', 'SeaPressure', 'Temp']]

    res = pred_sta.merge(gts_sta, left_on=['Station_Id', 'PredTime'], right_on=['stationID', 'Time'])
    res = res[['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime',
               'Direction_x', 'Speed_x', 'T2', 'SLP',
               'seapre_predict', 'temp_predict', 'dir_predict', 'speed_predict',
               'Direction_y', 'Speed_y', 'SeaPressure', 'Temp']]
    res.dropna(axis=0, how='any', inplace=True)
    res = res[res['seapre_predict'] > 400]

    wrf_seapre_rmse = np.sqrt(sklearn.metrics.mean_squared_error(res['SLP'] * 100, res['SeaPressure']))
    wrf_temp_rmse = np.sqrt(sklearn.metrics.mean_squared_error(res['T2'], res['Temp']))
    wrf_dir_rmse = np.sqrt(sklearn.metrics.mean_squared_error(res['Direction_x'], res['Direction_y']))
    wrf_speed_rmse = np.sqrt(sklearn.metrics.mean_squared_error(res['Speed_x'], res['Speed_y']))

    prediction_seapre_rmase = np.sqrt(sklearn.metrics.mean_squared_error(res['seapre_predict'], res['SeaPressure']))
    prediction_temp_rmse = np.sqrt(sklearn.metrics.mean_squared_error(res['temp_predict'], res['Temp']))
    prediction_dir_rmse = np.sqrt(sklearn.metrics.mean_squared_error(res['dir_predict'], res['Direction_y']))
    prediction_speed_rmse = np.sqrt(sklearn.metrics.mean_squared_error(res['speed_predict'], res['Speed_y']))

    print('RMSE of WRF[ 海平面气压: {0}, 气温: {1}, 风向: {2}, 风速: {3}'.format(str(wrf_seapre_rmse), str(wrf_temp_rmse),
                                                                      str(wrf_dir_rmse),
                                                                      str(wrf_speed_rmse)))
    print('RMSE of NEW[ 海平面气压: {0}, 气温: {1}, 风向: {2}, 风速: {3}'.format(str(prediction_seapre_rmase),
                                                                      str(prediction_temp_rmse),
                                                                      str(prediction_dir_rmse),
                                                                      str(prediction_speed_rmse)))


if __name__ == "__main__":
    all_res_to_one()
    all_gts_to_one()
    analysis_predict()
