# coding:utf-8
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.layers import LSTM, Input, Add, Dense, Dropout, Activation, \
    Concatenate, ZeroPadding2D, BatchNormalization, Flatten, \
    Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization
from keras.models import Model, load_model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from keras import backend as K, Sequential


def lgb_model(X, y, model_name, scaler_Y):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression_l2',  # 目标函数
        'metric': {'l2_root'},  # 评估函数
        'num_leaves': 511,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbosity': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }
    y = np.reshape(y, (-1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.15)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_eval, label=y_eval, reference=lgb_train)
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=40,
                    verbose_eval=100)
    print('Save model...')
    gbm.save_model(model_name)
    y_pre = gbm.predict(X_test)
    y_pre_ori = scaler_Y.inverse_transform(np.reshape(y_pre, (-1, 1)))
    y_true_ori = scaler_Y.inverse_transform(np.reshape(y_test, (-1, 1)))
    print('RMSE of LightGBM: {}'.format(np.sqrt(mean_squared_error(y_pre_ori, y_true_ori))))
    return gbm


def simple_rnn(X, extra_X):
    X_input = Input(X)
    extra_input = Input(extra_X)

    X = LSTM(256)(X_input)
    X = Concatenate()([X, extra_input])
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.8)(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.8)(X)

    X = Dense(1)(X)
    model = Model(inputs=[X_input, extra_input], outputs=X)
    return model


def swish(x):
    return x * K.sigmoid(x)


def ann(X_in):
    X_input = Input(X_in)
    X = Dense(1024)(X_input)
    X = BatchNormalization()(X)
    X = Activation(swish)(X)

    X = Dense(512)(X_input)
    X = BatchNormalization()(X)
    X = Activation(swish)(X)

    X = Dense(256)(X)
    X = BatchNormalization()(X)
    X = Activation(swish)(X)

    X = Dense(128)(X)
    X = BatchNormalization()(X)
    X = Activation(swish)(X)

    X = Dense(32)(X)
    X = BatchNormalization()(X)
    X = Activation(swish)(X)
    # X = BatchNormalization()(X)
    # X = Dropout(0.2)(X)
    X = Dense(1)(X)
    model = Model(inputs=X_input, outputs=X)
    return model


def build_lstm():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    print(model.layers)
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model

