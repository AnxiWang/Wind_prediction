import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
from wrf_utils import *
from logger import log
import configparser
import os


def data_from_analysis_to_wrf(analysis_dir, result_dir, wrf_dir):
    ec_files = get_ec_files(analysis_dir)
    sorted_ec_files = sorted(ec_files)
    wrf_data = nc.Dataset(wrf_dir)
    lon_w = wrf_data['XLONG'][0].data
    lat_w = wrf_data['XLAT'][0].data

    for fn in sorted_ec_files:
        analysis_data = nc.Dataset(fn, 'r')
        lon_a = analysis_data['longitude'][:].data
        lat = analysis_data['latitude'][:].data
        u10 = analysis_data['u10']
        v10 = analysis_data['v10']
        t2m = analysis_data['t2m']
        msl = analysis_data['msl']
        d2m = analysis_data['d2m']
        lon = []
        for list in lon_a:
            if list > 180:
                list = list - 360
            lon.append(list)
        lon = lon[253:775]
        lat = lat[128:409]
        idx = np.meshgrid(np.array(range(0, len(lon))), np.array(range(0, len(lat))))
        print(lon)
        d = np.meshgrid(lon, lat)
        pd_a = pd.DataFrame({'ilon': idx[0].ravel(), 'ilat': idx[1].ravel(),
                             'lon': d[0].ravel(), 'lat': d[1].ravel()})

        res_ul0 = []
        for index in range(u10.shape[0]):
            log.logger.info("start to analyse u10 ,index is : {}".format(index))
            grid_data = griddata((d[0].ravel(), d[1].ravel()), u10[index, 128:409, 253:775].ravel(),
                                 (lon_w, lat_w), method='nearest')
            res_ul0.append(grid_data)

        res_vl0 = []
        for index in range(v10.shape[0]):
            log.logger.info("start to analyse v10 ,index is : {}".format(index))
            grid_data = griddata((d[0].ravel(), d[1].ravel()), v10[index, 128:409, 253:775].ravel(),
                                 (lon_w, lat_w), method='nearest')
            res_vl0.append(grid_data)

        res_t2m = []
        for index in range(t2m.shape[0]):
            log.logger.info("start to analyse t2m ,index is : {}".format(index))
            grid_data = griddata((d[0].ravel(), d[1].ravel()), t2m[index, 128:409, 253:775].ravel(),
                                 (lon_w, lat_w), method='nearest')
            res_t2m.append(grid_data)

        res_msl = []
        for index in range(msl.shape[0]):
            log.logger.info("start to analyse msl ,index is : {}".format(index))
            grid_data = griddata((d[0].ravel(), d[1].ravel()), msl[index, 128:409, 253:775].ravel(),
                                 (lon_w, lat_w), method='nearest')
            res_msl.append(grid_data)

        res_q2 = []
        for index in range(msl.shape[0]):
            log.logger.info("start to analyse q2  ,index is : {}".format(index))
            q2 = dp_to_q2(d2m[index, 128:409, 253:775], msl[index, 128:409, 253:775])
            grid_data = griddata((d[0].ravel(), d[1].ravel()), q2.ravel(),
                                 (lon_w, lat_w), method='nearest')
            res_q2.append(grid_data)

        np.savez(result_dir + '/' + fn[-7:-3] + '_' + 'grid_data', u10=res_ul0, v10=res_vl0, t2m=res_t2m, msl=res_msl, q2=res_q2)
        log.logger.info("result is output to :{} ".format(result_dir + '/' + fn[-7:-3] + '_' + 'grid_data'))


if __name__ == "__main__":
    conf = configparser.ConfigParser()
    conf.read(os.getcwd() + '/config/config.ini', encoding="utf-8")
    analysis_dir = conf.get('data_dirs', 'analysis_dir')
    ear_grid_on_wrf = conf.get('data_dirs', 'ear_grid_on_wrf')
    wrf_dir = conf.get('data_dirs', 'wrf_dir')
    data_from_analysis_to_wrf(analysis_dir, ear_grid_on_wrf, wrf_dir)







