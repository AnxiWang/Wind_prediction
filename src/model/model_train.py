import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn import preprocessing
from DeepFM import DeepFM
import pickle
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.utils import shuffle

ignore_cols = []
numeric_cols = []
pick_cols = [u"竞拍地区", u"品牌id", u"车系id", u"车型id", u"车辆年款", u"车辆里程", u"车辆等级", u"车辆地区",
             u"客户地区", u"环保等级", u"车辆颜色", u"车辆使用类型", u"国系", u"是否热门车", u"是否常用车",
             u"成交月份", "car_age", "chassis_drive", "2_46_4261", "2_911_4292", "time_to_now", "safe_child","loss"]
y_col = [u"loss"]
drop_cols = [u"loss"]
USE_VALID = 1   #是否使用验证集
USE_PICK_COLS = 1

if USE_VALID:
    train_percent = 0.9
#模型参数定义
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
rounds = 50

def feature_preprocess(dftrain, train_add_feature):
    dftrain = dftrain.loc[:, pick_cols]
    dftrain = pd.concat([dftrain, train_add_feature], axis=1)
    dftrain = shuffle(dftrain)
    y_dftrain = dftrain["loss"].values.tolist()
    dftrain.drop(drop_cols, axis=1, inplace=True)
    return dftrain, y_dftrain

def feature_dictionary(dftrain):
    # 添加全零值以应对字典异常值
    dftrain.loc[len(dftrain), :] = 0

    feed_dict = {}
    tc = 0
    for col in dftrain.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            # map to a single index
            feed_dict[col] = tc
            tc += 1
        else:
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


def xgboost_model_train(dftrain, model_name):
    y_dftrain = dftrain["loss"].values.tolist()
    dftrain = dftrain.loc[:, pick_cols]
    # dftrain.drop(drop_cols, axis=1, inplace=True)

    for f in dftrain.columns:
        if dftrain[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(dftrain[f].values))
            dftrain[f] = lbl.transform(list(dftrain[f].values))

    dftrain.fillna((-999), inplace=True)
    dftrain = np.array(dftrain)
    dftrain = dftrain.astype(float)

    xgtrain = xgb.DMatrix(dftrain, label=y_dftrain)
    xgtrain_test = xgb.DMatrix(dftrain)
    xgbmodel = xgb.train(params=xgb_params, dtrain=xgtrain, num_boost_round=rounds)
    xgbmodel.save_model(model_name)

    y_train_leaf = xgbmodel.predict(xgtrain_test, pred_leaf=True)
    train_add_feature = pd.DataFrame(data=y_train_leaf)

    for i in range(0, rounds):
        train_add_feature[i] = round(train_add_feature[i] / 10, 0)

    return train_add_feature


def deepfm_training(dftrain, dict_file_name, train_add_feature):
    # 打乱顺序
    dftrain, y_train = feature_preprocess(dftrain, train_add_feature)

    dict_train = dftrain.copy()
    feed_dict, feed_dim = feature_dictionary(dict_train)

    # 保存编码字典
    dict_file = open(dict_file_name, "wb")
    pickle.dump(feed_dict, dict_file)
    dict_file.close()

    # 编码预处理
    xi_train, xv_train = data_parse(dftrain, feed_dict)

    dfm_params["feature_size"] = feed_dim
    dfm_params["field_size"] = len(xi_train[0])

    _get = lambda x, l: [x[i] for i in l]
    if USE_VALID:
        split_flag = int(train_percent * len(dftrain))
        train_idx = [i for i in range(split_flag)]
        valid_idx = [i for i in range(split_flag + 1, len(dftrain))]
    else:
        train_idx = [i for i in range(len(dftrain))]

    if USE_VALID:
        Xi_train_, Xv_train_, y_train_ = _get(xi_train, train_idx), _get(xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(xi_train, valid_idx), _get(xv_train, valid_idx), _get(y_train, valid_idx)
    else:
        Xi_train_, Xv_train_, y_train_ = _get(xi_train, train_idx), _get(xv_train, train_idx), _get(y_train, train_idx)

    dfm = DeepFM(**dfm_params)

    if USE_VALID:
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
    else:
        dfm.fit(Xi_train_, Xv_train_, y_train_)


if __name__ == "__main__":
    dftrain = pd.read_csv("/Users/anxi/Documents/Chebao/Gujia5.0/inputdata/train.csv")
    dftest = pd.read_csv("/Users/anxi/Documents/Chebao/Gujia5.0/inputdata/test.csv")

    xgb_model_name = "/Users/anxi/Documents/Chebao/Gujia5.0/model/xgb.model"
    dict_file_name = "/Users/anxi/Documents/Chebao/Gujia5.0/model/feed_dict.pkl"

    train_add_feature = xgboost_model_train(dftrain, xgb_model_name)
    print("=====xgboost train finish=====")

    deepfm_training(dftrain, dict_file_name, train_add_feature)
    print('=====deepfm train finish=====')


