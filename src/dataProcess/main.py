from wrfDataProcess import gatherGTS,gatherMonthGTS, dumpWRF, meshWRFtoStation, \
    buildDataset, dumpMonthWRF, meshMonthWRFtoStation, buildMonthDataset
from windFeature import readGTS
from dataProcessUtils import build_mesh, get_wrf_time, find_mesh_city_station
import pandas as pd
id_mesh_path = '/home/shared_data/external/IDWRF/202.108.199.14/IDWRF/OUTPUT_P/PACK_IDWRF' \
               '/pwrfout_d01.2019010100_2019010812.nc'
pc_mesh_path = '/home/shared_data/Wind_WRF/Data2/PACK_PCWRF/pack.pwrfout_d01.2017070700_2017071412.nc'

GTSPath = '/home/shared_data/Wind_WRF/Data1/GTS_OUT'
station_dir = '../../data/station/cityStation_drop.csv'

wrf_dir = '/home/shared_data/external/IDWRF/202.108.199.14/IDWRF/OUTPUT_P/PACK_IDWRF_6h/'
wrf_month_dir = '/home/shared_data/external/IDWRF/202.108.199.14/IDWRF/OUTPUT_P/PACK_IDWRF/'

pcwrf_dir = '/home/shared_data/Wind_WRF/Data1/PACK_PCWRF_6h/'
pcwrf_2017_dir = '/home/shared_data/Wind_WRF/Data2/PACK_PCWRF/'
pcwrf_2018_dir = '/home/shared_data/Wind_WRF/Data2/2018/'
# find the near mesh to the station
pc_mesh = pd.read_csv('../../data/output/pc_mesh.csv', encoding='utf-8')
station = pd.read_csv(station_dir, encoding='utf-8')
near_mesh = find_mesh_city_station(station, pc_mesh)
near_mesh.to_csv('../../data/output/near_mesh.csv', index=False)
# build mesh
# pc_mesh = build_mesh(pc_mesh_path)
# pc_mesh.to_csv('../../data/output/pc_mesh.csv')
# id_mesh = build_mesh(id_mesh_path)
# id_mesh.to_csv('../../data/output/id_mesh.csv')

# get station information
# station = pd.read_csv(station_dir, encoding='windows-1252')
# station = station[['stationID', 'LONG', 'LAT']] \
#     .groupby("stationID") \
#     .head(1) \
#     .reset_index() \
#     .drop(columns='index')

# get time label from WRF data
# wrf_times = get_wrf_time(wrf_dir).sort_values()
# wrf_month_times = get_wrf_time(wrf_month_dir).sort_values()

pcwrf_times = get_wrf_time(pcwrf_dir).sort_values()
pcwrf_2017_times = get_wrf_time(pcwrf_2017_dir).sort_values()
pcwrf_2018_times = get_wrf_time(pcwrf_2018_dir).sort_values()

if __name__ == "__main__":
    # # 印度洋数据处理
    # for year in range(2013, 2020, 1):
    #       out_name = 'wrf'
    #     # dataSetPath = GTSPath + '/' + str(i)      # 从GTS数据中提取需要的数据
    #     # readGTS(dataSetPath, str(i))
    #     # WRF: 2013-2017 6h output one value
    #     #      2018-2019 1h output one value
    #     if year < 2018:
    #         GTS = gatherGTS(year)
    #         wrf_pred = dumpWRF(near_mesh, wrf_dir, wrf_times, out_name, year)
    #         WRF = meshWRFtoStation(wrf_pred, station, mesh, year)
    #         dataset = buildDataset(year)
    #     if year >= 2018:
    #         # GTS = gatherGTS(year)
    #         for month in range(1, 13, 1):
    #             if month < 10:
    #                 month = '0' + str(month)
    #             else:
    #                 month = str(month)
    #             GTS = gatherMonthGTS(month, year)
    #             wrf_pred = dumpMonthWRF(near_mesh, wrf_month_dir, wrf_month_times, month, year)
    #             WRF = meshMonthWRFtoStation(wrf_pred, station, mesh, month, year)
    #             dataset = buildMonthDataset(month, year)
    # 太平洋数据处理
    for year in range(2014, 2019, 1):
        out_name = 'pcwrf'
        if year < 2017:
            # GTS = gatherGTS(year)
            pcwrf_pred = dumpWRF(near_mesh, pcwrf_dir, pcwrf_times, out_name, year)
        if year == 2017:
            pcwrf_pred = dumpWRF(near_mesh, pcwrf_2017_dir, pcwrf_2017_times, out_name, year)
            # for month in range(1, 13, 1):
            #     if month < 10:
            #         month = '0' + str(month)
            #     else:
            #         month = str(month)
            # pcwrf_pred = dumpMonthWRF(near_mesh, pcwrf_2017_dir, pcwrf_2017_times, out_name, month, year)
        if year == 2018:
            pcwrf_pred = dumpWRF(near_mesh, pcwrf_2018_dir, pcwrf_2018_times, out_name, year)
            # for month in range(1, 13, 1):
            #     if month < 10:
            #         month = '0' + str(month)
            #     else:
            #         month = str(month)
            # pcwrf_pred = dumpMonthWRF(near_mesh, pcwrf_2018_dir, pcwrf_2018_times, out_name, month, year)

