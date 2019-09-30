import os
import sys

sys.path.append('../dataProcess/')
import pandas as pd
import datetime
from dataProcessUtils import ymd, hours, get_wrf_time, as_str_h, as_str, build_mesh, find_nearest_point, \
    calculateSpeedFromUV
from sta_predict_utils import latest_wrf_date, load_model, get_wrf_path, get_sta_wrf_var, wrf_mesh_to_station
from tqdm import tqdm
import warnings
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
import pickle

warnings.filterwarnings("ignore")
import configparser

conf = configparser.ConfigParser()
conf.read('model_config.ini', encoding="utf-8")

wrf_dir = conf.get('data_dirs', 'test_wrf_dirs')
station_dir = conf.get('data_dirs', 'station_dir')
base_gts_dir = conf.get('data_dirs', 'GTSPath')
gts_pre_dir = conf.get('data_dirs', 'gts_pre_dir')
output_dir = conf.get('data_dirs', 'output_dir')
mesh_dir = conf.get('data_dirs', 'near_mesh')
id_mesh_dir = conf.get('data_dirs', 'id_mesh_dir')


ymdh = "%Y%m%d%H"
station = pd.read_csv(station_dir, encoding='utf-8')
near_mesh = pd.read_csv(mesh_dir, encoding='utf-8')
id_mesh = pd.read_csv(id_mesh_dir, encoding='utf-8')

features = ['Direction', 'Speed', 'UU', 'VV', 'Q2', 'T2', 'PSFC', 'QVAPOR', 'QCLOUD', 'HGT', 'RAINC', 'RAINNC',
            'SWDOWN', 'GSW', 'GLW', 'HFX', 'QFX', 'LH', 'TT', 'GHT', 'RH', 'SLP']


def do_predict():
    models = load_model()
    wrf_times = get_wrf_time(wrf_dir).sort_values()
    for each_time in tqdm(wrf_times, total=len(wrf_times)):
        year = each_time.strftime(ymdh)[0:4]

        file_name = '{0}/wrf_{1}_{2}.csv'.format(output_dir, as_str_h(each_time), as_str_h(each_time + hours(7 * 24 + 12)))
        output_file = "{0}/P_{1}_{2}_wind.csv".format(output_dir, as_str_h(each_time),
                                                      as_str_h(each_time + hours(7 * 24 + 12)))
        if not os.path.isfile(output_file):
            if os.path.isfile(file_name):
                sta_pred = pd.read_csv(file_name, encoding='utf-8')
                sta_list = sta_pred.Station_Id.unique()

                sta_data = [sta_pred[sta_pred.Station_Id == s] for s in tqdm(sta_list)]
                d = pd.DataFrame()

                cnt = 0
                for i in sta_list:
                    m = None
                    try:
                        m = models[i]
                    except:
                        pass
                    if m is not None:
                        if min(sta_data[cnt][features].count()) < 1:
                            print('There is no data for station {0}, skipping...'.format(i))
                        else:
                            each_sta = sta_data[cnt]
                            y = m.predict(sta_data[cnt][features])
                            y_prediction = pd.DataFrame(y, columns=['seapre_predict', 'temp_predict', 'dir_predict', 'speed_predict'])
                            each_sta['seapre_predict'] = y_prediction.seapre_predict.values
                            each_sta['temp_predict'] = y_prediction.temp_predict.values
                            each_sta['dir_predict'] = y_prediction.dir_predict.values
                            each_sta['speed_predict'] = y_prediction.speed_predict.values
                            d = pd.concat([d, each_sta])
                    cnt = cnt + 1
                d.to_csv(output_file)
            else:
                each_wrf = get_sta_wrf_var(wrf_dir, near_mesh, each_time, each_time + hours(7 * 24 + 12), year)
                if len(each_wrf) == 172:
                    wrf_meshed = wrf_mesh_to_station(station, each_wrf, id_mesh)
                    sta_data = calculateSpeedFromUV(wrf_meshed)
                    # drop a line with four NAN
                    sta_data['d'] = sta_data['Direction'].isnull() * 1 + sta_data['Speed'].isnull() * 1 + sta_data['T2'].isnull() * 1 + + sta_data['SLP'].isnull() * 1
                    sta_data = sta_data[sta_data['d'] < 4]
                    sta_data.to_csv(file_name, encoding='utf-8', index=0)


if __name__ == "__main__":
    # year = None
    # try:
    #     year = int(sys.argv[1])
    # except:
    #     logging.error('You must provide a valid lead argument. Example: python predict.py 2019')
    #     sys.exit()
    do_predict()
