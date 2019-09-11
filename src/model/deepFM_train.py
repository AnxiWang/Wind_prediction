import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
pick_cols = ['Direction_x', 'Speed_x', 'SeaPressure', 'StaPressure', 'P3', 'Temp', 'DPT']
target = ['Direction_y', 'Speed_y']
drop_cols = ['Direction_y', 'Speed_y']


def feature_preprocess(dftrain, train_add_feature):
    dftrain = dftrain.loc[:, pick_cols]
    dftrain = pd.concat([dftrain, train_add_feature], axis=1)
    dftrain = shuffle(dftrain)
    y_dftrain = dftrain["Direction_y"].values.tolist()
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


def deepfm_training(dftrain, dict_file_name, train_add_feature):
    # 打乱顺序
    dftrain, y_train = feature_preprocess(dftrain, train_add_feature)
    print(dftrain)
    print(y_train)
    exit()

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
    dataset = pd.read_csv('../../data/output/dataset_2013.csv', encoding='utf-8')
    [train, test] = train_test_split(dataset, test_size=0.2)

    xgb_model = 'xgb.model'

    train_add_feature = xgboost_model_train(train, xgb_model)
    print(train_add_feature)
    deepfm_training(train, train_add_feature)

