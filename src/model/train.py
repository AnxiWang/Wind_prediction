import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from model import *

import warnings
warnings.filterwarnings(action='ignore')

dataset = pd.read_csv('../../data/output/dataset_2013.csv', encoding='utf-8')

features = ['Direction_x',
            'Speed_x',
            'SeaPressure',
            'StaPressure',
            'P3',
            'Temp',
            'DPT'
            ]

target = ['Direction_y', 'Speed_y']

X = dataset[features]
y = np.array([dataset.Direction_y, dataset.Speed_y]).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


def train_lstm(train_x, train_y, test_x, test_y):
    # paras = list(map(lambda x: x.ravel(),
    #                  np.meshgrid(range(10, 200, 5), range(4, 30, 2))))
    # for lstm_layers, dense_layers in zip(paras[0], paras[1]):
    model = build_lstm()
    try:
        model.fit(train_x, train_y, batch_size=512, nb_epoch=30, validation_split=0.1)
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size,))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    print(predict)
    print(test_y)
    return predict, test_y


def randomForest_train(X_train, X_test, y_train, y_test):
    nrow = len(X_train)
    if nrow < 300:
        exit()

    paras = list(map(lambda x: x.ravel(),
                     np.meshgrid(range(10, 200, 5), range(4, 30, 2))))
    for ne, md in zip(paras[0], paras[1]):
        clf = MultiOutputRegressor(RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=200))

        clf.fit(X_train, y_train)

        y_multirf = clf.predict(X_test)

        # WRF预报的风向和风速
        y_wrf = np.array([X_test.Direction_x, X_test.Speed_x]).T
        y_WRF = pd.DataFrame(y_wrf, columns=['direction', 'speed'])
        # GTS观测风向和风速
        # y_gts = np.array([y_test.Direction_y, y_test.Speed_y]).T
        y_GTS = pd.DataFrame(y_test, columns=['direction', 'speed'])
        # 订正后的风向和风速
        y_prediction = pd.DataFrame(y_multirf, columns=['direction', 'speed'])

        wrf_dir_rmse = np.sqrt(mean_squared_error(y_WRF.direction, y_GTS.direction))
        wrf_speed_rmse = np.sqrt(mean_squared_error(y_WRF.speed, y_GTS.speed))

        prediction_dir_rmse = np.sqrt(mean_squared_error(y_prediction.direction, y_GTS.direction))
        prediction_speed_rmse = np.sqrt(mean_squared_error(y_prediction.speed, y_GTS.speed))

        print('n_estimators(ne): {0}, max_depth(md): {1}!'.format(ne, md))

        print('wrf direction rmse: {0}, wrf speed rmse: {1} '.format(wrf_dir_rmse, wrf_speed_rmse))
        print('prediction direction rmse: {0}, prediction speed rmse: {1}'.format(prediction_dir_rmse, prediction_speed_rmse))
        print('====================================================')

randomForest_train(X_train, X_test, y_train, y_test)
# train_lstm(X_train, X_test, y_train, y_test)
