import os
import re
import pandas as pd
from dataProcessUtils import *
from dateutil.relativedelta import relativedelta

# GTSPath = '../../data/GTS/2015'
from numba import jit

GTSPath = '/home/shared_data/Wind_WRF/Data1/GTS_OUT'
# stationPath = '/home/wanganxi/DataProcess/data'
# stationPathLocal = '../../data/station'
storePath = '../../data/station/'


# 获取GTS文件夹下面的所有满足需求的文件
@jit
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
                    getStationInfo(targetFilePath, outFileName, year)


# 针对每一个GTS文件提取站点风向和风速信息
@jit
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


if __name__ == "__main__":
    # 提取当前年份
    # year = time.strftime('%Y', time.localtime(time.time()))
    for i in range(2013, 2020, 1):
        dataSetPath = GTSPath + '/' + str(i)
        readGTS(dataSetPath, str(i))
        # windSetPath = storePath + str(i)
    #     get6hWindData('../../wind/GTS.out_20130505_wind.csv')
    # readGTS(GTSPath)

# def reduceDupStation(csvPath):
#     reducedStationDf = pd.DataFrame(columns=['stationID', 'LONG', 'LAT'])
#     outFilePath = stationPath + '/station_reduced.csv'
#     stationFiles = os.listdir(csvPath)
#     for eachFile in stationFiles:
#         eachFilePath = csvPath + '/' + eachFile
#         data = pd.read_csv(eachFilePath, encoding='windows-1252')
#         data = data.drop_duplicates()
#         reducedStationDf = reducedStationDf.append(data, ignore_index=True)
#     reducedStationDf = reducedStationDf.drop_duplicates()
#     reducedStationDf.to_csv(outFilePath, index=False, encoding='windows-1252')
#
#
# def getIndianStationInfo(csvPath):
#     stationInfoPath = csvPath + '/station_reduced.csv'
#     IndianStation = csvPath + '/IndianStation.csv'
#     stationDf = pd.read_csv(stationInfoPath, encoding='windows-1252')
#
#     # 根据经纬度范围进行初步删选
#     IndianStationDf = stationDf[(stationDf.LONG > 40) & (stationDf.LONG < 105) &
#                                 (stationDf.LAT > 0) & (stationDf.LAT < 30) &
#                                 (stationDf['stationID'].str.isdigit())]
#     IndianStationDf.to_csv(IndianStation, index=False, encoding='windows-1252')
