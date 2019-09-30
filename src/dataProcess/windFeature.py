import numpy as np
import time
import os
import re
import pandas as pd
from dataProcessUtils import *
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool
GTSPath = '/home/shared_data/Wind_WRF/Data1/GTS_OUT'
storePath = '../../data/wind/'

station_path = '../../data/station/sta_43.csv'
indianStationDf = pd.read_csv(station_path, encoding='windows-1252')


# 获取GTS文件夹下面的所有满足需求的文件
def readGTS(dataSetPath, year):
    files = os.listdir(dataSetPath)
    for file in files:
        if re.match(u'[0-9]{6}', file):
            eachYearPath = dataSetPath + '/' + file
            outFilePath = storePath + year + '/' + file
            if not os.path.exists(outFilePath):
                os.makedirs(outFilePath)
            print(eachYearPath)
            print('===========================')
            eachYearFiles = os.listdir(eachYearPath)
            for target in eachYearFiles:
                if re.match(u'GTS.out_[0-9]{8}', target):
                    targetFilePath = eachYearPath + '/' + target
                    outFileName = outFilePath + '/' + target + '_wind.csv'
                    print(targetFilePath)
                    if not os.path.isfile(outFileName):
                        getWindInfo(targetFilePath, outFileName, year)


# 针对每一个GTS文件提取站点风向和风速信息
def getWindInfo(targetFilePath, outFileName, year):
    if int(year) < 2018:
        stationDf = pd.DataFrame(
            columns=['stationID', 'LONG', 'LAT', 'Date', 'Hour', 'Direction', 'Speed', 'SeaPressure', 'StaPressure', 'P3',
                     'Temp', 'DPT', 'Time'])

        with open(targetFilePath, encoding='windows-1252') as f:
            for line in f.readlines():
                try:
                    line = ' '.join(line.split())  # change multiple space to one
                    # 先删除行首空格，再把减号与前一个数据连接的断开，再以空格分割
                    line = line.lstrip().replace(' -', '-').replace('-', ' -').replace('-888888', 'NaN').split(' ')
                    stationId = line[0]
                    stationLong = line[1]
                    stationLat = line[2]
                    date = line[3]
                    hour = line[4]
                    if int(date) < 10:
                        date = '0' + date
                    if int(hour) < 10:
                        hour = '0' + hour
                    windDirection = line[8]
                    windSpeed = line[9]
                    seaPressure = line[10]
                    staPressure = line[11]
                    p3 = line[12]
                    temp = line[13]
                    dpt = line[15]
                    # 这里判断每个月第一天中包含上个月最后一天数据
                    # 如果文件中读出的日期和文件名中的日期差值大于1，则将月份数据减一
                    if int(date) - int(targetFilePath.split('_')[-1][6:8]) > 2:
                        obsTime = targetFilePath.split('_')[-1][0:6]
                        obsTime = as_str_h(ym(obsTime) - relativedelta(months=1))
                        obsTime = obsTime[0:6] + date + hour
                    else:
                        obsTime = targetFilePath.split('_')[-1][0:6] + date + hour
                    # print(stationId in indianStationDf['stationID'].values.astype(np.str))
                    # 只提取印度洋需要修正的67个站点的数据
                    if stationId in indianStationDf['stationID'].values.astype(np.str):
                        stationDf = stationDf.append([{'stationID': stationId,
                                                       'LONG': stationLong,
                                                       'LAT': stationLat,
                                                       'Date': date,
                                                       'Hour': hour,
                                                       'Direction': windDirection,
                                                       'Speed': windSpeed,
                                                       'SeaPressure': seaPressure,
                                                       'StaPressure': staPressure,
                                                       'P3': p3,
                                                       'Temp': temp,
                                                       'DPT': dpt,
                                                       'Time': obsTime}], ignore_index=True)
                except Exception as e:
                    print(e)
                continue
            stationDf.to_csv(outFileName, index=False, encoding='windows-1252')
    elif int(year) >= 2018:
        stationDf = pd.DataFrame(
            columns=['stationID', 'LONG', 'LAT', 'Date', 'Hour', 'Direction', 'Speed', 'SeaPressure', 'StaPressure',
                     'P3',
                     'Temp', 'DPT', 'Time'])

        with open(targetFilePath, encoding='windows-1252') as f:
            for line in f.readlines():
                try:
                    line = ' '.join(line.split())  # change multiple space to one
                    # 先删除行首空格，再把减号与前一个数据连接的断开，再以空格分割
                    line = line.lstrip().replace(' -', '-').replace('-', ' -').replace('-888888', 'NaN').split(' ')
                    stationId = line[0]
                    stationLong = line[3]
                    stationLat = line[4]
                    date = line[1]
                    hour = line[2]
                    if int(hour) < 10:
                        hour = '0' + hour
                    windDirection = line[8]
                    windSpeed = line[9]
                    seaPressure = line[10]
                    staPressure = line[11]
                    p3 = line[12]
                    temp = line[13]
                    dpt = line[15]
                    obsTime = date + hour
                    # print(stationId in indianStationDf['stationID'].values.astype(np.str))
                    # 只提取印度洋需要修正的67个站点的数据
                    if stationId in indianStationDf['stationID'].values.astype(np.str):
                        stationDf = stationDf.append([{'stationID': stationId,
                                                       'LONG': stationLong,
                                                       'LAT': stationLat,
                                                       'Date': date,
                                                       'Hour': hour,
                                                       'Direction': windDirection,
                                                       'Speed': windSpeed,
                                                       'SeaPressure': seaPressure,
                                                       'StaPressure': staPressure,
                                                       'P3': p3,
                                                       'Temp': temp,
                                                       'DPT': dpt,
                                                       'Time': obsTime}], ignore_index=True)
                except Exception as e:
                    print(e)
                continue
            stationDf.to_csv(outFileName, index=False, encoding='windows-1252')


def get6hWindInfo(targetFilePath, outFileName):
    file = pd.read_csv(targetFilePath, encoding='utf-8')
    res = pd.DataFrame(columns=['stationID', 'LONG', 'LAT', 'Time', 'Direction', 'Speed'])
    for index, row in file.iterrows():
        hour = int(row['Hour'])
        if hour % 6 == 0:
            # print(row['Hour'])
            res = res.append([{'stationID': row.stationID,
                                'LONG': row.LONG,
                                'LAT': row.LAT,
                                'Time': row.Time,
                                'Direction': row.Direction,
                                'Speed': row.Speed
                               }], ignore_index=True)
    res.to_csv(outFileName, index=False, encoding='utf-8')


def get6hWindData(windSetPath):
    files = os.listdir(windSetPath)
    for file in files:
        if re.match(u'[0-9]{6}', file):
            eachYearPath = windSetPath + '/' + file
            print(eachYearPath)
            eachYearFiles = os.listdir(eachYearPath)
            for target in eachYearFiles:
                targetFilePath = eachYearPath + '/' + target
                outFileName = storePath + target + '_6h_wind.csv'
                print(targetFilePath)
                get6hWindInfo(targetFilePath, outFileName)
                

# if __name__ == "__main__":
#
#     for i in range(2013, 2020, 1):
#         dataSetPath = GTSPath + '/' + str(i)
#         readGTS(dataSetPath, str(i))


def job1(z):
    return readGTS(z[0], z[1])


# GTS观测数据从2017年12月更换格式
if __name__ == "__main__":
    dataSetPath = []
    year = []
    for i in range(2013, 2020, 1):
        dataSetPath.append(GTSPath + '/' + str(i))
        year.append(str(i))

    param = zip(dataSetPath, year)
    with Pool(10) as p:
        p.map(job1, param)
