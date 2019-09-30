import pandas as pd
import re

data_dir = '../../data/output_all'
train_data_time = ['2013', '2014', '2015', '2016']
train_files = ["{0}/dataset_{1}_new.csv".format(data_dir, x) for x in train_data_time]
res = [pd.read_csv(f, index_col=0, encoding='utf-8') for f in train_files]
train_dataset = pd.concat(res)
print(train_dataset)
train_dataset.to_csv('../../data/output_all/train.csv', encoding='utf-8')
# test_data_time = []
# for i in range(1, 13, 1):
#     if i < 10:
#         m = '0' + str(i)
#     test_data_time.append('2018' + m)
#     # test_data_time.append('2019' + m)
# print(test_data_time)
# test_files = ["{0}/dataset_{1}.csv".format(data_dir, x) for x in test_data_time]
# try:
#     test_res = [pd.read_csv(f, index_col=0, encoding='utf-8') for f in test_files]
#     test_dataset = pd.concat(test_res)
#     print(test_dataset)
# except:
#     print('file does not exist!')
# 合成测试集
# import os
# items = os.listdir(data_dir)
# newlist = []
# for names in items:
#     if len(names) == 18:
#         newlist.append(names)
# test_files = ["{0}/{1}".format(data_dir, x) for x in newlist]
#
# test_res = [pd.read_csv(f, index_col=0, encoding='utf-8') for f in test_files]
# test_dataset = pd.concat(test_res)
# print(test_dataset)
# test_dataset.to_csv('../../data/test.csv', encoding='utf-8')
