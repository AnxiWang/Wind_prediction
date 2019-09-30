import pandas as pd
import numpy as np
from multiprocessing import Pool

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import warnings

warnings.filterwarnings(action='ignore')

dataset = pd.read_csv('../../data/output_all/dataset_2013.csv', encoding='utf-8')
stations = pd.read_csv('../../data/station/sta_43.csv', encoding='utf-8')

features = ['Direction_x', 'Speed_x', 'UU', 'VV', 'Q2', 'T2', 'PSFC', 'QVAPOR', 'QCLOUD', 'HGT', 'RAINC', 'RAINNC',
            'SWDOWN', 'GSW', 'GLW', 'HFX', 'QFX', 'LH', 'TT', 'GHT', 'RH', 'SLP']
# 'StaPressure', 'P3', 'DPT'
target = ['SeaPressure', 'Temp', 'Direction_y', 'Speed_y']


def randomForest_train(dataset):
    [train, test] = train_test_split(dataset, test_size=0.2, random_state=200)

    sid = train.Station_Id.iloc[0]
    X_train = train[features]
    X_test = test[features]

    y_train = np.array([train.SeaPressure, train.Temp, train.Direction_y, train.Speed_y]).T
    y_test = np.array([test.SeaPressure, test.Temp, test.Direction_y, test.Speed_y]).T

    nrow = len(X_train)
    if nrow < 300:
        exit()
    P = []
    d = pd.DataFrame({'Station_Id': test.Station_Id.values,
                      'XLONG': test.XLONG.values,
                      'XLAT': test.XLAT.values,
                      'SpotTime': test.SpotTime.values,
                      'PredTime': test.PredTime.values})

    attempt_classifiers = {}
    attempt_predict_res = {}
    paras = list(map(lambda x: x.ravel(),
                     np.meshgrid(range(10, 50, 10), range(5, 40, 5))))
    for ne, md in zip(paras[0], paras[1]):
        # print(ne, md)
        seapre_wrf_col = 'seapre_wrf_' + str(ne) + '_' + str(md)
        temp_wrf_col = 'temp_wrf_' + str(ne) + '_' + str(md)
        dir_wrf_col = 'direction_wrf_' + str(ne) + '_' + str(md)
        speed_wrf_col = 'speed_wrf_' + str(ne) + '_' + str(md)

        seapre_gts_col = 'seapre_gts_' + str(ne) + '_' + str(md)
        temp_gts_col = 'temp_gts_' + str(ne) + '_' + str(md)
        dir_gts_col = 'direction_gts_' + str(ne) + '_' + str(md)
        speed_gts_col = 'speed_gts_' + str(ne) + '_' + str(md)

        seapre_new_col = 'seapre_new_' + str(ne) + '_' + str(md)
        temp_new_col = 'temp_new_' + str(ne) + '_' + str(md)
        dir_new_col = 'direction_new_' + str(ne) + '_' + str(md)
        speed_new_col = 'speed_new_' + str(ne) + '_' + str(md)

        rs = 200
        clf = MultiOutputRegressor(RandomForestRegressor(n_estimators=ne, max_depth=md, random_state=rs), n_jobs=5)
        clf.fit(X_train, y_train)
        y_multirf = clf.predict(X_test)

        # # 打印特征重要性，取的0或者1是对应y的位置
        # for index in range(4):
        #     label = target[index]
        #     importance = clf.estimators_[index].feature_importances_
        #     indices = np.argsort(importance)[::-1]
        #     # print("----the importance of features and its importance_score------")
        #     j = 1
        #     features_names = []
        #     im_list = []
        #     for i in indices[0:22]:
        #         f_name = X_train.columns.values[i]
        #         # print(j, f_name, importance[i])
        #         features_names.append(X_train.columns.values[i])
        #         im_list.append(importance[i])
        #         j += 1
        #     draw_importance(features_names, im_list, label, ne, md, sid)

        # WRF预报的风向和风速
        y_wrf = np.array([X_test.SLP, X_test.T2, X_test.Direction_x, X_test.Speed_x]).T
        # y_wrf = np.array([X_test.Direction_wrf, X_test.Speed_wrf]).T
        y_WRF = pd.DataFrame(y_wrf, columns=[seapre_wrf_col, temp_wrf_col, dir_wrf_col, speed_wrf_col])
        # d = d.join(y_WRF)
        # GTS观测风向和风速
        # y_gts = np.array([y_test.Direction_y, y_test.Speed_y]).T
        y_GTS = pd.DataFrame(y_test, columns=[seapre_gts_col, temp_gts_col, dir_gts_col, speed_gts_col])
        # d = d.join(y_GTS)
        # 订正后的风向和风速
        y_prediction = pd.DataFrame(y_multirf, columns=[seapre_new_col, temp_new_col, dir_new_col, speed_new_col])
        # d = d.join(y_prediction)

        wrf_seapre_rmase = np.sqrt(mean_squared_error(y_WRF[seapre_wrf_col], y_GTS[seapre_gts_col]))
        wrf_temp_rmse = np.sqrt(mean_squared_error(y_WRF[temp_wrf_col], y_GTS[temp_gts_col]))
        wrf_dir_rmse = np.sqrt(mean_squared_error(y_WRF[dir_wrf_col], y_GTS[dir_gts_col]))
        wrf_speed_rmse = np.sqrt(mean_squared_error(y_WRF[speed_wrf_col], y_GTS[speed_gts_col]))

        prediction_seapre_rmase = np.sqrt(mean_squared_error(y_prediction[seapre_new_col], y_GTS[seapre_gts_col]))
        prediction_temp_rmse = np.sqrt(mean_squared_error(y_prediction[temp_new_col], y_GTS[temp_gts_col]))
        prediction_dir_rmse = np.sqrt(mean_squared_error(y_prediction[dir_new_col], y_GTS[dir_gts_col]))
        prediction_speed_rmse = np.sqrt(mean_squared_error(y_prediction[speed_new_col], y_GTS[speed_gts_col]))

        prediction_rmse = np.sqrt(mean_squared_error(y_prediction, y_GTS))

        P.append([ne, md, rs, prediction_rmse, wrf_seapre_rmase, wrf_temp_rmse, wrf_dir_rmse, wrf_speed_rmse,
                  prediction_seapre_rmase, prediction_temp_rmse, prediction_dir_rmse, prediction_speed_rmse])
        attempt_classifiers[ne, md, rs] = clf
        attempt_predict_res[ne, md, rs] = d

        # print('n_estimators(ne): {0}, max_depth(md): {1}!'.format(ne, md))
        # print('wrf direction rmse: {0}, wrf speed rmse: {1} '.format(wrf_dir_rmse, wrf_speed_rmse))
        # print('prediction direction rmse: {0}, prediction speed rmse: {1}'.
        #       format(prediction_dir_rmse, prediction_speed_rmse))
        # print('***********************************************************')
    # d.to_csv('../../data/predict.csv', encoding='utf-8')

    best = pd.DataFrame(P) \
        .rename(columns={0: "ne", 1: "md", 2: "rs",
                         3: "prediction_rmse",
                         4: "wrf_seapre_rmase",
                         5: "wrf_temp_rmse",
                         6: "wrf_dir_rmse",
                         7: "wrf_speed_rmse",
                         8: "prediction_seapre_rmase",
                         9: "prediction_temp_rmse",
                         10: "prediction_dir_rmse",
                         11: "prediction_speed_rmse"}) \
        .sort_values('prediction_rmse').head(1)
    bne = best['ne'].iloc[0]
    bmd = best['md'].iloc[0]
    brs = best['rs'].iloc[0]
    return [sid, attempt_classifiers[bne, bmd, brs],
            attempt_predict_res[bne, bmd, brs],
            best['wrf_seapre_rmase'].iloc[0],
            best['wrf_temp_rmse'].iloc[0],
            best['wrf_dir_rmse'].iloc[0],
            best['wrf_speed_rmse'].iloc[0],
            best['prediction_seapre_rmase'].iloc[0],
            best['prediction_temp_rmse'].iloc[0],
            best['prediction_dir_rmse'].iloc[0],
            best['prediction_speed_rmse'].iloc[0]]


def draw_importance(features, importances, label, ne, md, sid):
    out_path = '../../data/importance/sta{0}_{1}_ne{2}_md{3}.png'.format(str(sid), label, str(ne), str(md))
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), np.array(importances)[indices], color='b', align='center')
    plt.yticks(range(len(indices)), np.array(features)[indices])
    plt.xlabel('Relative Importance')
    plt.savefig(out_path)
    # plt.show()


if __name__ == "__main__":
    sta_list = dataset.Station_Id.unique()
    # [48806. 48917. 48916. 48802. 48803. 56778.]
    print("Spliting data by stations...")
    sta_data = [dataset[dataset.Station_Id == s] for s in sta_list]

    # start the simulation
    with Pool(4) as p:
        res = list(tqdm(p.imap(randomForest_train, sta_data), total=len(sta_data)))

    rmse = pd.DataFrame([[i[0], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10]] for i in res if i[0] != None],
                        columns=['Station_Id',
                                 'wrf_seapre_rmase',
                                 'wrf_temp_rmse',
                                 'wrf_dir_rmse',
                                 'wrf_speed_rmse',
                                 'prediction_seapre_rmase',
                                 'prediction_temp_rmse',
                                 'prediction_dir_rmse',
                                 'prediction_speed_rmse'])
    rmse['seapre_pct'] = (rmse['wrf_seapre_rmase'] - rmse['prediction_seapre_rmase'])
    rmse['temp_pct'] = (rmse['wrf_temp_rmse'] - rmse['prediction_temp_rmse'])
    rmse['dir_pct'] = (rmse['wrf_dir_rmse'] - rmse['prediction_dir_rmse'])
    rmse['speed_pct'] = (rmse['wrf_speed_rmse'] - rmse['prediction_speed_rmse'])
    rmse.to_csv('../../data/output/train_rmse.csv', encoding='utf-8')

    import datetime

    print("Dumping models ...")
    model_filename = "../../data/model/model-{0}_lead_7.pkl" \
        .format(datetime.datetime.now().strftime("%Y%m%d"))

    sta_models = dict([[i[0], i[1]] for i in res])
    with open(model_filename, "wb") as of:
        pickle.dump(sta_models, of)
