import os
import re
import pandas as pd
from dataProcessUtils import *
from dateutil.relativedelta import relativedelta
from multiprocessing import Pool


GTSPath = '/home/shared_data/Wind_WRF/Data1/GTS_OUT'
storePath = '../../data/station/'


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
                    outFileName = outFilePath + '/' + target + '_station.csv'
                    print(targetFilePath)
                    if not os.path.isfile(outFileName):
                        getStationInfo(targetFilePath, outFileName, year)


# 针对每一个GTS文件提取站点风向和风速信息
def getStationInfo(targetFilePath, outFileName, year):
    if int(year) < 2018:
        stationDf = pd.DataFrame(columns=['stationID', 'LONG', 'LAT'])

        with open(targetFilePath, encoding='windows-1252') as f:
            for line in f.readlines():
                try:
                    line = ' '.join(line.split())  # change multiple space to one
                    # 先删除行首空格，再把减号与前一个数据连接的断开，再以空格分割
                    line = line.lstrip().replace(' -', '-').replace('-', ' -').replace('-888888', 'NaN').split(' ')
                    stationId = line[0]
                    stationLong = line[1]
                    stationLat = line[2]

                    stationDf = stationDf.append([{'stationID': stationId,
                                                   'LONG': stationLong,
                                                   'LAT': stationLat}], ignore_index=True)
                except Exception as e:
                    print(e)
                continue
            stationDf.to_csv(outFileName, index=False, encoding='windows-1252')
    elif int(year) >= 2018:
        stationDf = pd.DataFrame(columns=['stationID', 'LONG', 'LAT'])

        with open(targetFilePath, encoding='windows-1252') as f:
            for line in f.readlines():
                try:
                    line = ' '.join(line.split())  # change multiple space to one
                    # 先删除行首空格，再把减号与前一个数据连接的断开，再以空格分割
                    line = line.lstrip().replace(' -', '-').replace('-', ' -').replace('-888888', 'NaN').split(' ')
                    stationId = line[0]
                    stationLong = line[3]
                    stationLat = line[4]

                    stationDf = stationDf.append([{'stationID': stationId,
                                                   'LONG': stationLong,
                                                   'LAT': stationLat}], ignore_index=True)
                except Exception as e:
                    print(e)
                continue
            stationDf.to_csv(outFileName, index=False, encoding='windows-1252')


def job1(z):
    return readGTS(z[0], z[1])


if __name__ == "__main__":
    dataSetPath = []
    year = []
    for i in range(2013, 2020, 1):
        dataSetPath.append(GTSPath + '/' + str(i))
        year.append(str(i))

    param = zip(dataSetPath, year)
    with Pool(10) as p:
        p.map(job1, param)
