import numpy as np
import time
import os
import re
import pandas as pd

GTSPath = '../../data/GTS/2015'
# stationPath = '/home/wanganxi/DataProcess/data'
# stationPathLocal = '../../data/station'
storePath = '../../data/wind/'

indianStationPath = '../../data/station/IndianStation.csv'
indianStationDf = pd.read_csv(indianStationPath, encoding='windows-1252')


# 获取GTS文件夹下面的所有满足需求的文件
def readGTS(dataSetPath):
    files = os.listdir(dataSetPath)
    for file in files:
        if re.match(u'[0-9]{6}', file):
            eachYearPath = dataSetPath + '/' + file
            print(eachYearPath)
            print('===========================')
            eachYearFiles = os.listdir(eachYearPath)
            for target in eachYearFiles:
                if re.match(u'GTS.out_[0-9]{8}', target):
                    targetFilePath = eachYearPath + '/' + target
                    outFileName = storePath + target + '_wind.csv'
                    print(targetFilePath)
                    getWindInfo(targetFilePath, outFileName)


# 针对每一个GTS文件提取站点风向和风速信息
def getWindInfo(targetFilePath, outFileName):
    stationDf = pd.DataFrame(columns=['stationID', 'LONG', 'LAT', 'Date', 'Hour', 'Direction', 'Speed', 'Time'])
    try:
        with open(targetFilePath, encoding='windows-1252') as f:
            for line in f.readlines():
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
                obsTime = targetFilePath.split('_')[-1][0:6] + date + hour
                print(stationId + ' ' + windDirection + ' ' + windSpeed + ' ' + date + ' ' + hour + ' ' + obsTime)
                print(stationId in indianStationDf['stationID'].values.astype(np.str))
                if stationId in indianStationDf['stationID'].values.astype(np.str):
                    stationDf = stationDf.append([{'stationID': stationId,
                                                   'LONG': stationLong,
                                                   'LAT': stationLat,
                                                   'Date': date,
                                                   'Hour': hour,
                                                   'Direction': windDirection,
                                                   'Speed': windSpeed,
                                                   'Time': obsTime}], ignore_index=True)

            stationDf.to_csv(outFileName, index=False, encoding='windows-1252')
    except Exception as e:
        print(str(e))


if __name__ == "__main__":

    # 提取当前年份
    # year = time.strftime('%Y', time.localtime(time.time()))
    # 从GTS结果中提取站点信息（包括台站号和经纬度）
    from multiprocessing import Pool

    # with Pool(20) as p:
    # for i in range(2013, 2019, 1):
    #     dataSetPath = GTSPath + '/' + str(i)
    #     readGTS(dataSetPath)
    readGTS(GTSPath)
