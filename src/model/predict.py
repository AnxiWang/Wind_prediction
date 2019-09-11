import csv
import time
import warnings
import threading
from datetime import datetime, timedelta
import pandas as pd
import datetime as dt
from netCDF4 import Dataset
import datetime as dt  # Python standard library datetime  module
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from pandas import concat
from pandas.tseries.offsets import Hour
import multiprocessing as mp
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from data_preprocess import *
from pandas.tseries.offsets import Hour
from sklearn.metrics import mean_squared_error
import h5py
import multiprocessing as mp
from utils import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from models import *
from sklearn.externals import joblib
from pred_utils import *

pd.options.mode.chained_assignment = None
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from datetime import timedelta
import argparse
import configparser
import sys
import warnings

warnings.filterwarnings("ignore")


def do_temperature_predict(lead, last_obs_time=None):
    # 从配置文件中读取各种路径
    conf = configparser.ConfigParser()
    conf_path = os.getenv('PREDICT_CONF_DIR')
    if not conf_path:
        conf_path = '.'
    print(conf_path)
    conf.read('{0}/config.ini'.format(conf_path), encoding="utf-8")
    # 获取EC数据路径
    ec_dir = conf.get('data_dirs', 'ec_dir')
    # 获取处理后的EC数据路径
    ec_pre_dir = conf.get('data_dirs', 'ec_pre_dir')
    # 获取OBS数据路径
    obs_dir = conf.get('data_dirs', 'obs_dir')
    # 获取处理后的OBS数据路径
    obs_pre_dir = conf.get('data_dirs', 'obs_pre_dir')
    # 输出预测温度的文件路径
    output_dir = conf.get('data_dirs', 'output_dir')
    # 站点数据路径
    station_info_path = conf.get('data_dirs', 'station_info_path')
    # 站点聚类文件路径
    cluster_info_path = conf.get('data_dirs', 'cluster_info_path')
    # 需要预测的站点信息文件路径
    stations_predict_path = conf.get('data_dirs', 'stations_need_to_predict_path')
    # EC模式数据计算完经纬度之后的文件
    ec_mesh_path = 'data/ec_mesh.csv'

    # 读取需要订正预测结果的站点信息
    file = open(stations_predict_path, 'r')
    station_id_list = file.read().split(',')
    station_id_list = [s.strip() for s in station_id_list]
    # 读取EC模式中站点的经纬度信息、站点信息、站点聚类信息
    ec_mesh = pd.read_csv(ec_mesh_path)
    s_info = pd.read_csv('data/all_stations.csv', dtype={'Station_Id': str})
    cluster_info = pd.read_pickle('data/station_cluster_100.pickle')

    # change start time
    # last_ec_time, last_ec_dir = get_last_ec(ec_dir)
    # get_last_obs_time等函数来自pred_utils.py文件
    if last_obs_time is None:
        last_obs_time = get_last_obs_time(obs_dir)
    print(last_obs_time)
    # 读取OBS数据
    OBS = read_obs_data(last_obs_time, obs_dir)
    # 根据读取的OBS数据的日期去获取对应EC数据的路径并判断路径下是否有EC文件
    ec_dir_12 = os.path.join(ec_dir, last_obs_time.strftime('%Y%m%d12'))
    print(last_obs_time)
    # exit()
    ec_dir_00 = os.path.join(ec_dir, last_obs_time.strftime('%Y%m%d00'))
    last_ec_dir = None
    # last_ec_dir = ec_dir_00
    if os.path.exists(ec_dir_12):
        last_ec_dir = ec_dir_12
    elif os.path.exists(ec_dir_00):
        last_ec_dir = ec_dir_00
    if last_ec_dir == None:
        print('EC file not found!')
        os._exit()

    # 标签特征，也就是我们需要订正的预测温度
    label_feat = 'Temperature'
    # EC数据需要的特征
    need_feat = '2t'
    # OBS观测数据需要的特征列表，包括：站点ID，时间，温度、10min风速、相对湿度和露点温度
    feat_list = ['Station_Id', 'Time', 'Temperature', 'Wd_10Min', 'Relative_Humidity', 'Tem_Dew_Point']
    # change lead
    lead_day_list = [i for i in range(1, lead + 1)]
    # print(lead_day_list)

    # result_dir=output_dir + "/{}".format(last_obs_time.strftime('%Y%m%d%H'))

    # if not os.path.exists(result_dir):
    #    os.makedirs(result_dir, mode=0o777)
    pool = Pool(processes=48)  # 创建48个进程
    results = []

    for lead_day in range(1, lead + 1):
        pre_sta_t = last_obs_time + timedelta(hours=3) + timedelta(days=lead_day - 1)
        pre_ed_t = last_obs_time + timedelta(days=lead_day)
        pre_time_range = pd.date_range(pre_sta_t, pre_ed_t, freq='3H')
        # 一次循环只读取了一天的数据
        EC = read_ec_data(last_ec_dir, pre_time_range, need_feat)
        # print(pre_time_range)

        predict_len = lead_day * 24
        time_len = lead_day * 8
        # 预测结果列表，主要包括站点ID， 开始时间，结束时间，日平均气温，日最高温和日最低温。
        # res_df = pd.DataFrame(
        #     columns=['station_id', 'start_time', 'end_time', 'Day_Ave_Temperature', 'Day_Max_Temperature',
        #              'Day_Min_Temperature'])
        # cnt = 0

        # 如何在这里开启cpu多核进行模型加载和预测
        # print(len(station_id_list))
        # job1 = threading.Thread(target=prediction, name="job1",
        # args=())
        # with Pool(2) as p:
        # res_df, save_path = p.apply_async(prediction_station, (OBS, EC, s_info,
        # ec_mesh, last_obs_time, pre_time_range, station_id_list, cluster_info, lead_day, label_feat, feat_list,
        # predict_len, time_len, output_dir))

        results.append(pool.apply_async(prediction_station,
                                        (OBS, EC, s_info, ec_mesh, last_obs_time, pre_time_range, station_id_list,
                                         cluster_info, lead_day, label_feat, feat_list, predict_len, time_len,
                                         output_dir,)))
        for res in results:
            print(res.get())

        # res_df, save_path = prediction_station(OBS, EC, s_info, ec_mesh, last_obs_time, pre_time_range, station_id_list,
        #                                        cluster_info, lead_day, label_feat, feat_list, predict_len, time_len, output_dir)
        # save_path = '{0}/t_{1}_{2}_TP24H.csv'.format(output_dir, last_obs_time.strftime('%Y%m%d%H'),
        #                                              pre_ed_t.strftime('%Y%m%d%H'))
        # writeList2CSV(results, save_path)

        # with open(save_path, 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        # for row in results:
        #     csvfile.write(row.get())
        #     csvfile.write("/n")
        # results_pd = pd.DataFrame(results)
        # results_pd.to_csv(save_path)
        print('Prediction {} day Done.'.format(lead_day))
        results = []
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕


def writeList2CSV(myList, filePath):
    try:
        file = open(filePath, 'w')
        for items in myList:
            for item in items.get():
                file.write(item)
            file.write("\n")
    except Exception:
        print("数据写入失败，请检查文件路径及文件编码是否正确")
    finally:
        file.close();  # 操作完成一定要关闭


def prediction_station(OBS, EC, s_info, ec_mesh, last_obs_time, pre_time_range, station_id_list,
                       cluster_info, lead_day, label_feat, feat_list, predict_len, time_len, output_dir):
    res_df = pd.DataFrame(
        columns=['station_id', 'start_time', 'end_time', 'Day_Ave_Temperature', 'Day_Max_Temperature',
                 'Day_Min_Temperature'])
    cnt = 0
    for station_id in station_id_list:
        if len(cluster_info[cluster_info['Station_Id'] == station_id].reset_index()) == 0:
            print('Station Id: {} not in the cluster infomation'.format(station_id))
            continue
        station_cluster = int(
            cluster_info[cluster_info['Station_Id'] == station_id].reset_index().loc[0, 'cluster'])
        # D:\code\temperature\model\stationCluster({})_leadDay({})_label({})_predictLen({})_timeLen({})_scalerX.pkl

        # D:\code\temperature\model\stationCluster({})_leadDay({})_label({})_predictLen({})_timeLen({})_scalerY.pkl
        # scaler_X_filename = 'model\stationCluster({})_leadDay({})_label({})_predictLen({})_timeLen({})_scalerX.pkl' \
        #     .format(station_cluster, lead_day, label_feat, predict_len, time_len)
        scaler_X_filename = "/home/shared_data/external/production/code/temperature/model/stationCluster({})_leadDay({})_label({})_predictLen({})_timeLen({})_scalerX.pkl" \
            .format(station_cluster, lead_day, label_feat, predict_len, time_len)
        scaler_y_filename = "/home/shared_data/external/production/code/temperature/model/stationCluster({})_leadDay({})_label({})_predictLen({})_timeLen({})_scalerY.pkl" \
            .format(station_cluster, lead_day, label_feat, predict_len, time_len)
        # D:\code\temperature\model\lightgbm_stationCluster(2)_leadDay(6)_label(Temperature)_predictLen(144)_timeLen(48).txt
        lgb_model_name = '/home/shared_data/external/production/code/temperature/model/lightgbm_stationCluster({})_leadDay({})_label({})_predictLen({})_timeLen({}).txt' \
            .format(station_cluster, lead_day, label_feat, predict_len, time_len)
        # 生成用于预测的数据集合，包含OBS和EC数据
        X, t_range = build_dataset_using_df_for_predict(OBS, EC, s_info, ec_mesh, last_obs_time, pre_time_range,
                                                        lead_day,
                                                        station_id, predict_len, time_len, label_feat, feat_list,
                                                        drop_label_time=False, time_skip=1)
        # print(X)
        if X.shape[0] == 0:
            print('Station Id: {} not have value'.format(station_id))
            continue
        if not os.path.exists(lgb_model_name):
            print('LightGBM model for station id:{} not found, use EC prediction replace!'.format(station_id))
            y_pre_ori = X['EC_' + label_feat]
        else:
            # 加载模型
            gbm = lgb.Booster(model_file=lgb_model_name)
            # 加载数据
            # print(scaler_X_filename)
            # print(scaler_y_filename)
            scaler_X = joblib.load(scaler_X_filename)
            scaler_y = joblib.load(scaler_y_filename)
            numeric_feats = X.dtypes[(X.dtypes != "object") \
                                     & (X.dtypes != "datetime64[ns]") \
                                     & (X.dtypes != "datetime64")].index
            col_names = list(numeric_feats)

            X[col_names] = scaler_X.transform(X[col_names])
            X_ = pd.get_dummies(X, dummy_na=False).values

            y_pre = gbm.predict(X_)
            y_pre_ori = scaler_y.inverse_transform(np.reshape(y_pre, (-1, 1)))
        time_sta = t_range.iloc[0]
        time_ed = t_range.iloc[-1]
        max_t = np.max(y_pre_ori)
        min_t = np.min(y_pre_ori)
        ave_t = np.mean(y_pre_ori)
        res_df.loc[cnt] = [station_id, time_sta, time_ed, ave_t, max_t, min_t]
        cnt += 1
    save_path = '{0}/t_{1}_{2}_TP24H.csv'.format(output_dir, last_obs_time.strftime('%Y%m%d%H'),
                                                 time_sta.strftime('%Y%m%d%H'))
    res_df.to_csv(save_path)
    return res_df


if __name__ == "__main__":

    lead = None
    try:
        lead = int(sys.argv[1])
    except:
        logging.error('You must provide a valid lead argument. Example: ./predict.py 1')
        sys.exit()
    startTime = time.clock()
    print(startTime)
    do_temperature_predict(lead)
    endTime = time.clock()
    print(endTime)
    t = endTime - startTime
    print("Run time is ", t)
