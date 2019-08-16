from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

m = Basemap()
# m = Basemap(projection='cyl', lat_0=35,
#               lon_0=110, resolution='l')  # 实例化一个map
m.drawcoastlines()  # 画海岸线
m.drawmapboundary(fill_color='white')
m.drawcountries(linewidth=0.5)
# m.fillcontinents(color='white', lake_color='white')  # 画大洲，颜色填充为白色

parallels = np.arange(-90., 90., 10.)  # 这两行画纬度，范围为[-90,90]间隔为10
m.drawparallels(parallels, labels=[False, True, True, False])
meridians = np.arange(-180., 180., 20.)  # 这两行画经度，范围为[-180,180]间隔为20
m.drawmeridians(meridians, labels=[True, False, False, True])

stationdf = pd.read_csv('../../data/station/IndianStation.csv', encoding='windows-1252')
allStationLon = stationdf['LONG'].copy()
allStationLat = stationdf['LAT'].copy()
print(len(allStationLon))
print(len(allStationLat))

shipdf = stationdf[stationdf['stationID'].isin(['SHIP'])].copy()
shipLon = shipdf['LONG'].copy()
shipLat = shipdf['LAT'].copy()
print(len(shipLon))
print(len(shipLat))

diff_flag = [not f for f in stationdf['stationID'].isin(['SHIP'])]
comStationdf = stationdf[diff_flag]
comStationLon = comStationdf['LONG'].copy()
comStationLat = comStationdf['LAT'].copy()
print(len(comStationLon))
print(len(comStationLat))

# allLon, allLat = m(allStationLon, allStationLat)
shipLon, shipLat = m(shipLon, shipLat)    # lon, lat为给定的经纬度，可以使单个的，也可以是列表
comStationLon, comStationLat = m(comStationLon, comStationLat)

# m.scatter(allLon, allLat, s=5, c='coral')
m.scatter(shipLon, shipLat, c='g', s=2)# 标注出所在的点，s为点的大小，还可以选择点的性状和颜色等属性
m.scatter(comStationLon, comStationLat, c='coral', s=1)

plt.show()
