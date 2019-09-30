from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geopy.distance import geodesic


def find_nearest_point(p, grid):
    # 经纬度差值的平方和
    grid['dist'] = (grid['LONG'] - p[0]) ** 2 + (grid['LAT'] - p[1]) ** 2
    # # 真实距离，单位km
    # dist = []
    # for index in range(0, grid.shape[0], 1):
    #     d = geodesic((p[1], p[0]), (grid.loc[index].LAT, grid.loc[index].LONG)).km
    #     dist.append(d)
    # grid['dist(km)'] = pd.DataFrame(dist)
    # fun = grid['stationID'].apply(lambda x: x.isdigit())
    # grid = grid[fun == True]
    return grid.nsmallest(4, columns='dist')


m = Basemap()
# m = Basemap(projection='cyl', lat_0=35, lon_0=110, resolution='l')  # 实例化一个map
m.drawcoastlines(color='grey')  # 画海岸线
m.drawmapboundary(fill_color='white')
m.drawcountries(linewidth=0.5)
# m.fillcontinents(color='white', lake_color='white')  # 画大洲，颜色填充为白色

parallels = np.arange(-90., 90., 10.)  # 这两行画纬度，范围为[-90,90]间隔为10
m.drawparallels(parallels, labels=[False, True, True, False])
# m.drawparallels([-12.1947, 57.8061], color='r')
# m.drawmeridians([63.218475, 180], color='r')
m.drawparallels([-21.0524, 32.3064], color='b')
m.drawmeridians([28.7757, 131.224], color='b')
meridians = np.arange(-180., 180., 20.)  # 这两行画经度，范围为[-180,180]间隔为20
m.drawmeridians(meridians, labels=[True, False, False, True])

stationdf = pd.read_csv('../../data/station/station_reduced.csv', encoding='windows-1252')
citydf = pd.read_csv('../../data/station/city.csv', encoding='utf-8')
city_sta_df = pd.read_csv('../../data/station/sta_43.csv', encoding='utf-8')
nearSta = pd.DataFrame(columns=['stationID', 'LONG', 'LAT', 'dist'])
# for index in range(0, citydf.shape[0], 1):
#     s_lon = citydf.loc[index].long
#     s_lat = citydf.loc[index].lat
#     nn = find_nearest_point([s_lon, s_lat], stationdf)
#     # print(nn)
#     nearSta = nearSta.append([{'stationID': citydf.loc[index].code,
#     'LONG': citydf.loc[index].long, 'LAT': citydf.loc[index].lat}], ignore_index=True)
#     nearSta = nearSta.append(nn)
# nearSta.to_csv('../../data/station/cityStation_dist_20_all.csv', encoding='utf-8')
# 港口城市经纬度信息
cityLabel = citydf['code'].copy()
cityLon = citydf['long'].copy()
cityLat = citydf['lat'].copy()
# 港口城市最近的站点的经纬度信息
city_staLabel = city_sta_df['stationID'].copy()
city_sta_lon = city_sta_df['LONG'].copy()
city_sta_lat = city_sta_df['LAT'].copy()
# 所有站点经纬度信息
allStationLon = stationdf['LONG'].copy()
allStationLat = stationdf['LAT'].copy()
# 部分船只经纬度信息
shipdf = stationdf[stationdf['stationID'].isin(['SHIP'])].copy()
shipLon = shipdf['LONG'].copy()
shipLat = shipdf['LAT'].copy()
# 去除部分船只之后的站点经纬度信息
diff_flag = [not f for f in stationdf['stationID'].isin(['SHIP'])]
comStationdf = stationdf[diff_flag]
comStationLon = comStationdf['LONG'].copy()
comStationLat = comStationdf['LAT'].copy()

nearStaLon = nearSta['LONG'].copy()
nearStaLat = nearSta['LAT'].copy()


# allLon, allLat = m(allStationLon, allStationLat)
cityLon, cityLat = m(cityLon, cityLat)
city_sta_Lon, city_sta_Lat = m(city_sta_lon, city_sta_lat)
shipLon, shipLat = m(shipLon, shipLat)
comStationLon, comStationLat = m(comStationLon, comStationLat)
nearStaLon, nearStaLat = m(nearStaLon, nearStaLat)

# m.scatter(allLon, allLat, s=5, c='coral')
# m.scatter(shipLon, shipLat, c='g', s=2)# 标注出所在的点，s为点的大小，还可以选择点的性状和颜色等属性
# m.scatter(comStationLon, comStationLat, c='g', s=1)
m.scatter(cityLon, cityLat, c='red', s=30, label=cityLabel)
# m.scatter(nearStaLon, nearStaLat, c='g', s=2)
m.scatter(city_sta_Lon, city_sta_Lat, marker='x', c='yellow', s=20, label=city_staLabel)

plt.show()
