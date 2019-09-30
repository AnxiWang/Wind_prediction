import os
import sys

sys.path.append('../dataProcess/')
import pandas as pd
import datetime
from dataProcessUtils import ymd, hours, get_wrf_time, as_str_h, as_str, build_mesh, find_nearest_point, \
    calculateSpeedFromUV
from predict_utils import *
import warnings

warnings.filterwarnings("ignore")
import configparser

conf = configparser.ConfigParser()
conf.read('model_config.ini', encoding="utf-8")
wrf_dir = conf.get('data_dirs', 'test_wrf_dirs')
station_dir = conf.get('data_dirs', 'station_dir')
base_gts_dir = conf.get('data_dirs', 'GTSPath')
gts_pre_dir = conf.get('data_dirs', 'gts_pre_dir')
output_dir = conf.get('data_dirs', 'output_dir')

ymdh = "%Y%m%d%H"
station = pd.read_csv(station_dir, encoding='utf-8')
features = ['Direction_x', 'Speed_x', 'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT']


def do_wind_predict():
    # get the latest wrf date
    date_start = latest_wrf_date(wrf_dir)
    date_end = date_start + hours(7 * 24 + 12)
    print(as_str(date_start) + " and " + as_str(date_end))

    models = load_model()

    wrf_file_path = wrf_dir + '/' + get_wrf_path(date_start, date_end)
    print(wrf_file_path)
    wrf_pred = get_wrf_var(wrf_file_path, date_start, date_end)
    # print(wrf_pred)
    datelist = [as_str(date_start)]
    while date_start < date_end - hours(12):
        add_date = date_start + datetime.timedelta(days=1)
        datelist.append(as_str(add_date))
    # load_latest_gts(base_gts_dir, datelist)
    gts_tal = load_gts(gts_pre_dir, datelist)
    gts_tal.to_csv('../../data/gts.csv')

    mesh = build_mesh(wrf_file_path)

    sta_wind_pred = wrf_mesh_to_station(station, wrf_pred, mesh)
    WRF = calculateSpeedFromUV(sta_wind_pred)

    WRF.to_csv('../../data/wrf_predict.csv')
    # 重新读取一次是为了解决下面合并会出现类型冲突的问题
    GTS = pd.read_csv('../../data/gts.csv', encoding='utf-8')
    WRF = pd.read_csv('../../data/wrf_predict.csv', encoding='utf-8')
    # 6个小时输出一个
    for index, var in WRF.iterrows():
        if int(str(int(var['PredTime']))[8:10]) % 6 != 0:
            WRF.drop(index, inplace=True)
    sta_wind_gts_pred = WRF.merge(GTS, left_on=['Station_Id', 'PredTime'], right_on=['stationID', 'Time'])
    # build prediction dataset
    predict_dataset_path = '../../data/predict_dataset_' + as_str(date_start) + '-' + as_str(date_end) + '.csv'
    predict_dataset = pd.DataFrame(sta_wind_gts_pred,
                                   columns=['Station_Id', 'XLONG', 'XLAT', 'SpotTime', 'PredTime',
                                            'Direction_x', 'Speed_x',
                                            'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Direction_y',
                                            'Speed_y'])
    # fill nan
    predict_dataset.SeaPressure.fillna(predict_dataset.SeaPressure.mean(), inplace=True)
    predict_dataset.StaPressure.fillna(predict_dataset.StaPressure.mean(), inplace=True)
    predict_dataset.P3.fillna(predict_dataset.P3.mean(), inplace=True)
    predict_dataset.Temp.fillna(predict_dataset.Temp.mean(), inplace=True)
    predict_dataset.DPT.fillna(predict_dataset.DPT.mean(), inplace=True)
    predict_dataset.fillna(0, inplace=True)

    predict_dataset.to_csv(predict_dataset_path, encoding='utf-8')

    d = pd.DataFrame({'Station_Id': predict_dataset.Station_Id.values,
                      'XLONG': predict_dataset.XLONG.values,
                      'XLAT': predict_dataset.XLAT.values,
                      'SpotTime': predict_dataset.SpotTime.values,
                      'PredTime': predict_dataset.PredTime.values,
                      'dir_wrf': predict_dataset.Direction_x.values,
                      'speed_wrf': predict_dataset.Speed_x.values,
                      'dir_gts': predict_dataset.Direction_y.values,
                      'speed_gts': predict_dataset.Speed_y.values})

    y = models.predict(predict_dataset[features])

    y_prediction = pd.DataFrame(y, columns=['dir_predict', 'speed_predict'])
    d = d.join(y_prediction)

    output_file = "{0}/P_{1}_{2}_wind.csv".format(output_dir, as_str(date_start), as_str(date_end))
    d.drop_duplicates(inplace=True)
    d.to_csv(output_file)


if __name__ == "__main__":
    do_wind_predict()
