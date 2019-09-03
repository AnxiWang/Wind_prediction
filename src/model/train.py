import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


import warnings
warnings.filterwarnings(action='ignore')

dataset = pd.read_csv('../../data/output/dataset_fill_drop.csv', encoding='utf-8')

features = ['Direction_x',
            'Speed_x',
            'SeaPressure',
            'StaPressure',
            'P3',
            'Temp',
            'DPT'
            ]

target = ['Direction_y', 'Speed_y']


def station_train(data):
    nrow = data.shape[0]
    if nrow < 300:
        return [None, None, None, None, None]
    train, test = train_test_split(data, test_size=0.2)

    paras = list(map(lambda x: x.ravel(),
                     np.meshgrid(range(10, 200, 5), range(4, 30, 2))))
    for ne, md in zip(paras[0], paras[1]):
        clf = MultiOutputRegressor(RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=200), n_jobs=-1)

        y = np.array([train.Direction_y, train.Speed_y]).T
        clf.fit(train[features], y)

        y_multirf = clf.predict(test[features])

        # WRF预报的风向和风速
        y_wrf = np.array([test.Direction_x, test.Speed_x]).T
        y_WRF = pd.DataFrame(y_wrf, columns=['direction', 'speed'])
        # GTS观测风向和风速
        y_gts = np.array([test.Direction_y, test.Speed_y]).T
        y_GTS = pd.DataFrame(y_gts, columns=['direction', 'speed'])
        # 订正后的风向和风速
        y_prediction = pd.DataFrame(y_multirf, columns=['direction', 'speed'])

        # y_prediction.to_csv('pd_data.csv')

        wrf_dir_rmse = np.sqrt(mean_squared_error(y_WRF.direction, y_GTS.direction))
        wrf_speed_rmse = np.sqrt(mean_squared_error(y_WRF.speed, y_GTS.speed))

        prediction_dir_rmse = np.sqrt(mean_squared_error(y_prediction.direction, y_GTS.direction))
        prediction_speed_rmse = np.sqrt(mean_squared_error(y_prediction.speed, y_GTS.speed))

        print('n_estimators(ne): {0}, max_depth(md): {1}!'.format(ne, md))

        print('wrf direction rmse: {0}, wrf speed rmse: {1} '.format(wrf_dir_rmse, wrf_speed_rmse))
        print('prediction direction rmse: {0}, prediction speed rmse: {1}'.format(prediction_dir_rmse, prediction_speed_rmse))


if __name__ == '__main__':
    station_train(dataset)





