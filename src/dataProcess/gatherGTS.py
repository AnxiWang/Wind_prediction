import numpy as np
import os
import re
import pandas as pd
from dataProcessUtils import *
from dateutil.relativedelta import relativedelta

GTSPath = '../../data/GTS'
storePath = '../../data/GTS/'

indianStationPath = '../../data/station/IndianStation.csv'
indianStationDf = pd.read_csv(indianStationPath, encoding='windows-1252')


# 获取GTS文件夹下面的所有满足需求的文件
def readGTS_TTAA(dataSetPath):
    files = os.listdir(dataSetPath)
    for file in files:
        if re.match(u'GTS.out_TTAA_[0-9]{10}', file):
            print(file)
            eachFile = dataSetPath + '/' + file
            outFileName = storePath + file + '_wind.csv'

            getWindInfo(eachFile, outFileName)
            print('===========================')
            # eachYearFiles = os.listdir(eachYearPath)
            # for target in eachYearFiles:
            #     if re.match(u'GTS.out_[0-9]{8}', target):
            #         targetFilePath = eachYearPath + '/' + target
            #         outFileName = outFilePath + '/' + target + '_wind.csv'
            #         print(targetFilePath)
            #         getWindInfo(targetFilePath, outFileName)


# 针对每一个GTS文件提取站点风向和风速信息
def getWindInfo(targetFilePath, outFileName):
    stationDf = pd.DataFrame(
        columns=['stationID', 'LONG', 'LAT', 'Time', 'Direction', 'Speed', 'SeaPressure', 'StaPressure',
                 'Temp', 'DPT'])

    with open(targetFilePath, encoding='windows-1252') as f:
        for line in f.readlines():
            try:
                line = ' '.join(line.split())  # change multiple space to one
                # 先删除行首空格，再把减号与前一个数据连接的断开，再以空格分割
                line = line.lstrip().replace(' -', '-').replace('-', ' -').replace('-888888', 'NaN').split(' ')
                stationId = line[1]
                stationLong = line[2]
                stationLat = line[3]
                time = line[0]
                windDirection = line[10]
                windSpeed = line[9]
                seaPressure = line[5]
                staPressure = line[5]
                # p3 = line[12]
                temp = float(line[7]) + 273.15
                dpt = line[8]
                # print(stationId in indianStationDf['stationID'].values.astype(np.str))
                # 只提取印度洋需要修正的67个站点的数据
                if stationId in indianStationDf['stationID'].values.astype(np.str):
                    stationDf = stationDf.append([{'stationID': stationId,
                                                   'LONG': stationLong,
                                                   'LAT': stationLat,
                                                   'Time': time,
                                                   'Direction': windDirection,
                                                   'Speed': windSpeed,
                                                   'SeaPressure': seaPressure,
                                                   'StaPressure': staPressure,
                                                   'Temp': temp,
                                                   'DPT': dpt}], ignore_index=True)
            except Exception as e:
                print(e)
            continue
        stationDf.to_csv(outFileName, index=False, encoding='windows-1252')


if __name__ == "__main__":
    readGTS_TTAA(GTSPath)

    # for i in range(2013, 2019, 1):
    #     dataSetPath = GTSPath + '/' + str(i)
    #     readGTS(dataSetPath, str(i))


