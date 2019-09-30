import pandas as pd
import configparser

from wrfDataProcess import gatherGTS,gatherMonthGTS, dumpWRF, meshWRFtoStation, \
    buildDataset, dumpMonthWRF, meshMonthWRFtoStation, buildMonthDataset
from windFeature import readGTS
from dataProcessUtils import build_mesh, get_wrf_time, find_mesh_city_station

conf = configparser.ConfigParser()
conf.read('config.ini', encoding="utf-8")
id_mesh_path = conf.get('data_dirs', 'id_mesh_path')
pc_mesh_path = conf.get('data_dirs', 'pc_mesh_path')
GTSPath = conf.get('data_dirs', 'GTSPath')
wrf_dir = conf.get('data_dirs', 'wrf_dir')
wrf_month_dir = conf.get('data_dirs', 'wrf_month_dir')
pcwrf_dir = conf.get('data_dirs', 'pcwrf_dir')
pcwrf_2017_dir = conf.get('data_dirs', 'pcwrf_2017_dir')
pcwrf_2018_dir = conf.get('data_dirs', 'pcwrf_2018_dir')

station_dir = conf.get('data_dirs', 'station_dir')
pc_mesh_dir = conf.get('data_dirs', 'pc_mesh_dir')
id_mesh_dir = conf.get('data_dirs', 'id_mesh_dir')
sta_near_mesh = conf.get('data_dirs', 'sta_near_mesh')

# build mesh()
# pc_mesh = build_mesh(pc_mesh_path)
# pc_mesh.to_csv(pc_mesh_dir, encoding='utf-8')
# id_mesh = build_mesh(id_mesh_path)
# id_mesh.to_csv(id_mesh_dir, encoding='utf-8')

pc_mesh = pd.read_csv(pc_mesh_dir, encoding='utf-8')
id_mesh = pd.read_csv(id_mesh_dir, encoding='utf-8')
# get station information
station = pd.read_csv(station_dir, encoding='utf-8')
# find the near mesh to the station
near_mesh = find_mesh_city_station(station, id_mesh)
near_mesh.to_csv(sta_near_mesh, index=False)

# get time label from WRF data
wrf_times = get_wrf_time(wrf_dir).sort_values()
wrf_month_times = get_wrf_time(wrf_month_dir).sort_values()

pcwrf_times = get_wrf_time(pcwrf_dir).sort_values()
pcwrf_2017_times = get_wrf_time(pcwrf_2017_dir).sort_values()
pcwrf_2018_times = get_wrf_time(pcwrf_2018_dir).sort_values()

if __name__ == "__main__":
    # 印度洋数据处理
    for year in range(2013, 2018, 1):
        out_name = 'wrf'
        # dataSetPath = GTSPath + '/' + str(i)      # 从GTS数据中提取需要的数据
        # readGTS(dataSetPath, str(i))
        # WRF: 2013-2017 6h output one value
        #      2018-2019 1h output one value
        if year < 2018:
            GTS = gatherGTS(year)
            wrf_pred = dumpWRF(near_mesh, wrf_dir, wrf_times, out_name, year)
            WRF = meshWRFtoStation(wrf_pred, station, id_mesh, out_name, year)
            dataset = buildDataset(out_name, year)
        if year >= 2018:
            GTS = gatherGTS(year)
            wrf_pred = dumpWRF(near_mesh, wrf_month_dir, wrf_month_times, out_name, year)
            WRF = meshWRFtoStation(wrf_pred, station, id_mesh, out_name, year)
            dataset = buildDataset(out_name, year)
            # GTS = gatherGTS(year)
            # for month in range(1, 13, 1):
            #     if month < 10:
            #         month = '0' + str(month)
            #     else:
            #         month = str(month)
            #     GTS = gatherMonthGTS(month, year)
            #     wrf_pred = dumpMonthWRF(near_mesh, wrf_month_dir, wrf_month_times, month, year)
            #     WRF = meshMonthWRFtoStation(wrf_pred, station, id_mesh, out_name, month, year)
            #     dataset = buildMonthDataset(month, year)
    # # 太平洋数据处理
    # for year in range(2013, 2019, 1):
    #     out_name = 'pcwrf'
    #     if year < 2017:
    #         # GTS = gatherGTS(year)
    #         pcwrf_pred = dumpWRF(near_mesh, pcwrf_dir, pcwrf_times, out_name, year)
    #         PCWRF = meshWRFtoStation(pcwrf_pred, station, near_mesh, out_name, year)
    #         dataset = buildDataset(out_name, year)
    #
    #     if year == 2017:
    #         pcwrf_pred = dumpWRF(near_mesh, pcwrf_2017_dir, pcwrf_2017_times, out_name, year)
    #         # for month in range(1, 13, 1):
    #         #     if month < 10:
    #         #         month = '0' + str(month)
    #         #     else:
    #         #         month = str(month)
    #         # pcwrf_pred = dumpMonthWRF(near_mesh, pcwrf_2017_dir, pcwrf_2017_times, out_name, month, year)
    #     if year == 2018:
    #         pcwrf_pred = dumpWRF(near_mesh, pcwrf_2018_dir, pcwrf_2018_times, out_name, year)
    #         # for month in range(1, 13, 1):
    #         #     if month < 10:
    #         #         month = '0' + str(month)
    #         #     else:
    #         #         month = str(month)
    #         # pcwrf_pred = dumpMonthWRF(near_mesh, pcwrf_2018_dir, pcwrf_2018_times, out_name, month, year)

