import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from DeepFM import DeepFM

rounds = 50
xgb_params = {
    'min_child_weight': 20,
    'eta': 0.1,
    'colsample_bytree': 0.8,
    'max_depth': 6,
    'subsample': 0.8,
    'alpha': 1,
    'gamma': 0.2,
    'silent': 1,
    'verbose_eval': True,
    'seed': 0,
    'eval_metric': "logloss",
}
# 模型参数定义
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 128,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.relu,   #tanh, relu
    "epoch": 20,
    "batch_size": 1024,
    "learning_rate": 0.002,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2018,
    "loss_type": "logloss"  # "logloss" or "mse", "exp-loss"
}
pick_cols = ['Direction_x', 'Speed_x', 'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT', 'Direction_y', 'Speed_y']
target = ['Direction_y', 'Speed_y']
drop_cols = ['Direction_y', 'Speed_y']


def feature_preprocess(dftrain, train_add_feature):
    dftrain = dftrain.loc[:, pick_cols]
    dftrain = pd.concat([dftrain, train_add_feature], axis=1)
    dftrain = dftrain.dropna(axis=0, how='any')
    dftrain = shuffle(dftrain)
    y_dftrain = dftrain['Direction_y'].values.tolist()
    dftrain.drop(drop_cols, axis=1, inplace=True)
    return dftrain, y_dftrain


def feature_dictionary(dftrain):
    # 添加全零值以应对字典异常值
    dftrain.loc[len(dftrain), :] = 0

    feed_dict = {}
    tc = 0
    for col in dftrain.columns:
        us = dftrain[col].unique()
        feed_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
        tc += len(us)
    feed_dim = tc
    return feed_dict, feed_dim


def data_parse(dftrain, feed_dict):
    dfv = dftrain.copy()
    for col in dftrain.columns:
        dftrain[col] = dftrain[col].map(feed_dict[col])
        dfv[col] = 1.

    xi = dftrain.values.tolist()
    xv = dfv.values.tolist()

    return xi, xv


def xgboost_model_train(train, model_name):
    y_train = train['Direction_y'].values.tolist()
    dftrain = train.loc[:, pick_cols]

    # Encode labels with value between 0 and n_classes-1
    for f in dftrain.columns:
        if dftrain[f].dtype == 'float64':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(dftrain[f].values))
            dftrain[f] = lbl.transform(list(dftrain[f].values))

    dftrain = np.array(dftrain)
    dftrain = dftrain.astype(float)

    xgtrain = xgb.DMatrix(dftrain, label=y_train)
    xgtrain_test = xgb.DMatrix(dftrain)
    xgbmodel = xgb.train(params=xgb_params, dtrain=xgtrain, num_boost_round=rounds)
    xgbmodel.save_model(model_name)

    y_train_leaf = xgbmodel.predict(xgtrain_test, pred_leaf=True)
    train_add_feature = pd.DataFrame(data=y_train_leaf)

    for i in range(0, rounds):
        train_add_feature[i] = round(train_add_feature[i] / 10, 0)
    return train_add_feature


def deepfm_training(dftrain, train_add_feature):
    # 打乱顺序
    dftrain, y_train = feature_preprocess(dftrain, train_add_feature)

    dict_train = dftrain.copy()
    feed_dict, feed_dim = feature_dictionary(dict_train)

    # 保存编码字典
    # dict_file = open(dict_file_name, "wb")
    # pickle.dump(feed_dict, dict_file)
    # dict_file.close()

    # 编码预处理
    xi_train, xv_train = data_parse(dftrain, feed_dict)

    dfm_params["feature_size"] = feed_dim
    dfm_params["field_size"] = len(xi_train[0])

    _get = lambda x, l: [x[i] for i in l]

    train_idx = [i for i in range(len(dftrain))]
    Xi_train_, Xv_train_, y_train_ = _get(xi_train, train_idx), _get(xv_train, train_idx), _get(y_train, train_idx)

    dfm = DeepFM(**dfm_params)
    dfm.fit(Xi_train_, Xv_train_, y_train_)


if __name__ == "__main__":
    dataset = pd.read_csv('../../data/output/dataset_2013.csv', encoding='utf-8')
    [train, test] = train_test_split(dataset, test_size=0.2)

    xgb_model = 'xgb.model'

    train_add_feature = xgboost_model_train(train, xgb_model)
    # print(train_add_feature)
    deepfm_training(train, train_add_feature)

