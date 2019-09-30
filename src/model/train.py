import pandas as pd
import numpy as np
from multiprocessing import Pool
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.filterwarnings(action='ignore')

dataset = pd.read_csv('../../data/output/train.csv', encoding='utf-8')
# era_dataset = pd.read_csv('../../data/output/wrf_gts_era_2013.csv', encoding='utf-8')

features = ['Direction_x', 'Speed_x', 'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT']
era_feature = ['Direction_wrf',
               'Speed_wrf',
               'SeaPressure',
               'StaPressure',
               'P3',
               'Temp',
               'DPT',
               'Direction_era',
               'Speed_era',
               'MSL',
               'T2M']
target = ['Direction_y', 'Speed_y']
wrf_target = ['Direction_x', 'Speed_x']
era_target = ['Direction_gts', 'Speed_gts']


# Station_Id,XLONG,XLAT,SpotTime,PredTime,Direction_y,Speed_y

# X = dataset[features]
# y = np.array([dataset.Direction_y, dataset.Speed_y]).T
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


def randomForest_train(dataset):
    # X = dataset[features]
    # # X = dataset[era_feature]
    # y = np.array([dataset.Direction_y, dataset.Speed_y]).T
    # # y = np.array([dataset.Direction_gts, dataset.Speed_gts]).T
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    [train, test] = train_test_split(dataset, test_size=0.2, random_state=200)
    X_train = train[features]
    X_test = test[features]

    y_train = np.array([train.Direction_y, train.Speed_y]).T
    y_test = np.array([test.Direction_y, test.Speed_y]).T

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
                     np.meshgrid(range(10, 200, 10), range(5, 40, 5))))
    for ne, md in zip(paras[0], paras[1]):
        dir_wrf_col = 'direction_wrf_' + str(ne) + '_' + str(md)
        speed_wrf_col = 'speed_wrf_' + str(ne) + '_' + str(md)
        dir_gts_col = 'direction_gts_' + str(ne) + '_' + str(md)
        speed_gts_col = 'speed_gts_' + str(ne) + '_' + str(md)
        dir_new_col = 'direction_new_' + str(ne) + '_' + str(md)
        speed_new_col = 'speed_new_' + str(ne) + '_' + str(md)
        rs = 200
        clf = MultiOutputRegressor(RandomForestRegressor(n_estimators=ne, max_depth=md, random_state=rs), n_jobs=10)
        clf.fit(X_train, y_train)
        y_multirf = clf.predict(X_test)

        # # 打印特征重要性
        # importance = clf.estimators_[0].feature_importances_
        # indices = np.argsort(importance)[::-1]
        # print("----the importance of features and its importance_score------")
        # j = 1
        # features_names = []
        # im_list = []
        # for i in indices[0:11]:
        #     f_name = X_train.columns.values[i]
        #     print(j, f_name, importance[i])
        #     features_names.append(X_train.columns.values[i])
        #     im_list.append(importance[i])
        #     j += 1
        # draw_importance(features_names, im_list)

        # WRF预报的风向和风速
        y_wrf = np.array([X_test.Direction_x, X_test.Speed_x]).T
        # y_wrf = np.array([X_test.Direction_wrf, X_test.Speed_wrf]).T
        y_WRF = pd.DataFrame(y_wrf, columns=[dir_wrf_col, speed_wrf_col])
        d = d.join(y_WRF)
        # GTS观测风向和风速
        # y_gts = np.array([y_test.Direction_y, y_test.Speed_y]).T
        y_GTS = pd.DataFrame(y_test, columns=[dir_gts_col, speed_gts_col])
        d = d.join(y_GTS)
        # 订正后的风向和风速
        y_prediction = pd.DataFrame(y_multirf, columns=[dir_new_col, speed_new_col])
        d = d.join(y_prediction)

        wrf_dir_rmse = np.sqrt(mean_squared_error(y_WRF[dir_wrf_col], y_GTS[dir_gts_col]))
        wrf_speed_rmse = np.sqrt(mean_squared_error(y_WRF[speed_wrf_col], y_GTS[speed_gts_col]))

        prediction_dir_rmse = np.sqrt(mean_squared_error(y_prediction[dir_new_col], y_GTS[dir_gts_col]))
        prediction_speed_rmse = np.sqrt(mean_squared_error(y_prediction[speed_new_col], y_GTS[speed_gts_col]))

        P.append([ne, md, rs, wrf_dir_rmse, wrf_speed_rmse, prediction_dir_rmse, prediction_speed_rmse])
        attempt_classifiers[ne, md, rs] = clf
        attempt_predict_res[ne, md, rs] = d

        print('n_estimators(ne): {0}, max_depth(md): {1}!'.format(ne, md))
        print('wrf direction rmse: {0}, wrf speed rmse: {1} '.format(wrf_dir_rmse, wrf_speed_rmse))
        print('prediction direction rmse: {0}, prediction speed rmse: {1}'.
              format(prediction_dir_rmse, prediction_speed_rmse))
        print('***********************************************************')
    d.to_csv('../../data/predict.csv', encoding='utf-8')

    best = pd.DataFrame(P) \
        .rename(columns={0: "ne", 1: "md", 2: "rs",
                         3: "wrf_dir_rmse",
                         4: "wrf_speed_rmse",
                         5: "prediction_dir_rmse",
                         6: "prediction_speed_rmse"}) \
        .sort_values('prediction_dir_rmse').head(1)
    bne = best['ne'].iloc[0]
    bmd = best['md'].iloc[0]
    brs = best['rs'].iloc[0]
    return [attempt_classifiers[bne, bmd, brs],
            attempt_predict_res[bne, bmd, brs],
            best['wrf_dir_rmse'].iloc[0],
            best['wrf_speed_rmse'].iloc[0],
            best['prediction_dir_rmse'].iloc[0],
            best['prediction_speed_rmse'].iloc[0]]


def draw_importance(features, importances):
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), np.array(importances)[indices], color='b', align='center')
    plt.yticks(range(len(indices)), np.array(features)[indices])
    plt.xlabel('Relative Importance')
    plt.show()


if __name__ == "__main__":
    res_rf = randomForest_train(dataset)

    import datetime

    print("Dumping models ...")
    model_filename = "../../data/model/model-{0}_lead_7.pkl" \
        .format(datetime.datetime.now().strftime("%Y%m%d"))

    with open(model_filename, "wb") as of:
        pickle.dump(res_rf[0], of)

    # res_resnet = resnet_train(dataset)
