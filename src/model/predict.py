import os
import sys
sys.path.append('../dataProcess/')
import pandas as pd
from netCDF4 import Dataset
import glob
import datetime as dt
import os.path
import pickle
import numpy as np
from dataProcessUtils import ymd, hours, get_wrf_time, as_str_h

import warnings
warnings.filterwarnings("ignore")
ymdh = "%Y%m%d%H"
wrf_dir = '/home/shared_data/external/IDWRF/202.108.199.14/IDWRF/OUTPUT_P/PACK_IDWRF'


def do_wind_predict():
    # conf = configparser.ConfigParser()
    # conf_path = os.getenv('PREDICT_CONF_DIR')
    # if not conf_path:
    #     conf_path = '.'
    #
    # conf.read('{0}/config.ini'.format(conf_path), encoding="utf-8")
    #
    # ec_dir = conf.get('data_dirs', 'ec_dir')
    # obs_dir = conf.get('data_dirs', 'obs_dir')
    # output_dir = conf.get('data_dirs', 'output_pre_dir')
    #
    # obs_times = get_obs_time(obs_dir).sort_values()
    # obs_times = obs_times[obs_times > ymd('20170101')]

    # get the latest wrf date
    start = latest_wrf_date(wrf_dir)
    end = start + hours(7 * 24 + 12)
    print(str(start)+" and "+str(end))

    models = load_model()

    wrf_file_path = wrf_dir + '/' + get_wrf_path(start, end)
    print(wrf_file_path)
    # 找到最新的WRF文件，未编写代码：根据wrf的预报截止时间找出所有的GTS数据
    wrf_pred = get_wrf_var(wrf_file_path, start, end)
    print(wrf_pred)
    exit()

    GTS = load_latest_gts(gts_dir, gts_times, 100)

    mesh = build_mesh()
    # print(mesh)
    logging.info("Mapping mesh data to stations...")
    sta_tp_pred = tp_mesh_to_station(OBS, tp_pred, mesh)

    logging.info("Merging forecast data with observation data...")

    sta_tp_obs_pred = \
        sta_tp_pred.merge(OBS, \
                          left_on=["SpotTime", "Station_Id"],
                          right_on=['Time', 'Station_Id']) \
            .drop(columns=['Time'])

    if max(sta_tp_obs_pred.count()) < 1:
        logging.error("No proper data for forecasting.")
        raise ValueError("No proper data for forecasting.")

    logging.info("Categorising Precipitation...")
    tp_col = [
        'Precipitation_24H',
        'Precipitation_12H',
        'Precipitation_6H',
        'Precipitation_3H',
        'Precipitation_1H',
        'Precipitation_24H_Target',
        'PredTP']
    for c in tp_col:
        if sta_tp_obs_pred.columns.contains(c):
            sta_tp_obs_pred[c] = \
                sta_tp_obs_pred[c] \
                    .apply(v_category_precipitation).astype('int32')

    sta_list = sta_tp_obs_pred.Station_Id.unique()
    logging.info("Spliting data by statoins...")
    sta_data = [sta_tp_obs_pred[sta_tp_obs_pred.Station_Id == s] for
                s in tqdm(sta_list)]

    logging.info("Start prediction models...")
    results = []
    cnt = 0
    for i in sta_list:
        m = None
        try:
            m = models[i]
        except:
            pass
        if (m != None):
            if min(sta_data[cnt][features].count()) < 1:
                logging.warning("There is no data for station {0}, skipping...".format(i))
            else:
                results.append([i, start, end, m.predict(prepare_feature(sta_data[cnt]))])
        cnt = cnt + 1

    R = pd.DataFrame(results).rename(
        columns={0: 'Station_Id',
                 1: 'SpotTime',
                 2: 'PredTime',
                 3: 'Precipitation_24H'})
    R['Precipitation_24H'] = R['Precipitation_24H'].map(lambda x: x[0])
    SpotTime = R.SpotTime[0]
    PredTime = R.PredTime[0]

    output_file = "{0}/p_{1}_{2}_TP24H.csv".format(output_dir, as_str(SpotTime), as_str(PredTime))
    R.to_csv(output_file)
    logging.info("Saved output to file {0}.".format(output_file))


def latest_wrf_date(wrf_dir):
    """
    :param wrf_dir: WRF数据存放路径，只获取到文件夹的名称，即日期
    :return: 最新的WRF数据日期
    """
    wrf_times = get_wrf_time(wrf_dir).sort_values()
    return pd.to_datetime(np.sort(wrf_times)[-1])


def get_wrf_path(start, end):
    return 'pwrfout_d01.{0}_{1}.nc'.format(as_str_h(start), as_str_h(end))


# load model
def load_model():
    # 根据预测时长读取对应的模型
    files = glob.glob("../../data/model/*.pkl")
    dates = [i[23:31] for i in files]
    latest = max(dates)
    model_filename = '../../data/model/model-{0}_lead_7.pkl'.format(latest)
    print(model_filename)
    with open(model_filename, 'rb') as of:
        return pickle.load(of)


def ymd_h(x):
    return pd.to_datetime(x, format="%Y%m%d%H")


# dump wrf data into one (2013-2017 by year)
def get_wrf_var(wrf_file_dir, start, end):
    res = []
    with Dataset(wrf_file_dir, mode='r', format='NETCDF4_CLASSIC') as ds:
        variable_keys = ds.variables.keys()
        # print(variable_keys)
        if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys and 'XLAT' in variable_keys:
            res.append([ymd_h(start).strftime(ymdh),
                        (ymd_h(end) - hours(2 * 24)).strftime(ymdh),
                        ds['XLONG'][:].data,
                        ds['XLAT'][:].data,
                        ds['U10'][:].data,
                        ds['V10'][:].data])
    return res



if __name__ == "__main__":
    do_wind_predict()
