import numpy as np
import time
import os
import re
import pandas as pd

GTSPath = '/home/shared_data/Wind_WRF/Data1/GTS_OUT'
stationPath = '/home/wanganxi/Wind/data/station'
stationPathLocal = '../../data/station'
storePath = '/home/wanganxi/Wind/data/GTS/'


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
                    eachYearFilePath = eachYearPath + '/' + target
                    outFileName = storePath + target + '.csv'
                    print(eachYearFilePath)
                    getStationInfo(eachYearFilePath, outFileName)


def getStationInfo(targetFilePath, outFileName):
    stationDf = pd.DataFrame(columns=['stationID', 'LONG', 'LAT'])
    try:
        with open(targetFilePath, encoding='windows-1252') as f:
            for line in f.readlines():
                line = ' '.join(line.split())  # change multiple space to one
                # 先删除行首空格，再把减号与前一个数据连接的断开，再以空格分割
                line = line.lstrip().replace(' -', '-').replace('-', ' -').split(' ')
                stationId = line[0]
                stationLong = line[1]
                stationLat = line[2]
                print(stationId + ' ' + stationLong + ' ' + stationLat)
                stationDf = stationDf.append([{'stationID': stationId, 'LONG': stationLong, 'LAT': stationLat}], ignore_index=True)
            stationDf.to_csv(outFileName, index=False, encoding='utf-8')
    except:
        print("some errors occur")


def reduceDupStation(csvPath):
    reducedStationDf = pd.DataFrame(columns=['stationID', 'LONG', 'LAT'])
    outFilePath = stationPath + '/station_reduced.csv'
    stationFiles = os.listdir(csvPath)
    for eachFile in stationFiles:
        eachFilePath = csvPath + '/' + eachFile
        data = pd.read_csv(eachFilePath, encoding='windows-1252')
        data = data.drop_duplicates()
        reducedStationDf = reducedStationDf.append(data, ignore_index=True)
    reducedStationDf = reducedStationDf.drop_duplicates()
    reducedStationDf.to_csv(outFilePath, index=False, encoding='windows-1252')


def getIndianStationInfo(csvPath):
    stationInfoPath = csvPath + '/station_reduced.csv'
    IndianStation = csvPath + '/IndianStation.csv'
    stationDf = pd.read_csv(stationInfoPath, encoding='windows-1252')

    # 根据经纬度范围进行初步删选
    IndianStationDf = stationDf[(stationDf.LONG > 40) & (stationDf.LONG < 105) &
                                (stationDf.LAT > 0) & (stationDf.LAT < 30) &
                                (stationDf['stationID'].str.isdigit())]
    IndianStationDf.to_csv(IndianStation, index=False, encoding='windows-1252')


if __name__ == "__main__":

    # 提取当前年份
    year = time.strftime('%Y', time.localtime(time.time()))
    # 从GTS结果中提取站点信息（包括台站号和经纬度）
    from multiprocessing import Pool

    # with Pool(20) as p:
    for i in range(2013, 2014, 1):
        dataSetPath = GTSPath + '/' + str(i)
        readGTS(dataSetPath)
    reduceDupStation(storePath)
    getIndianStationInfo(stationPath)
