import pickle

from tqdm import tqdm
import pandas as pd
import datetime as dt
import netCDF4 as nc
import os
import warnings

warnings.filterwarnings("ignore")
ymdh = "%Y%m%d%H"
wrf_pre_dir = '../../data/wrfout_pre'
wrf_dir = '/home/shared_data/external/IDWRF/202.108.199.14/IDWRF/OUTPUT_P/PACK_IDWRF_6h/'


def hours(h):
    return dt.timedelta(hours=h)


def ymd_h(x):
    return pd.to_datetime(x, format="%Y%m%d%H")


def dump_wrf_var(near_mesh, wrf_dir, spot, pred, year):
    global nc_name
    spot_str = spot.strftime(ymdh)
    pred_str = pred.strftime(ymdh)
    res = []
    for index, var in tqdm(near_mesh.iterrows(), total=near_mesh.shape[0]):
        for i in range(0, len(spot_str)):
            if spot_str[i][0:4] == year:
                if int(year) < 2018:
                    nc_name = '6h.pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                elif int(year) == 2018:
                    nc_name = 'pack.pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                elif int(year) > 2018:
                    nc_name = 'pwrfout_d01.' + spot_str[i] + '_' + pred_str[i] + '.nc'
                file = wrf_dir + nc_name
                # print(file)
                with nc.Dataset(file, mode='r', format='NETCDF4_CLASSIC') as ds:
                    variable_keys = ds.variables.keys()
                    if 'U10' in variable_keys and 'V10' in variable_keys and 'XLONG' in variable_keys \
                            and 'XLAT' in variable_keys and 'UU' in variable_keys and 'VV' in variable_keys \
                            and 'Q2' in variable_keys and 'PSFC' in variable_keys and 'QVAPOR' in variable_keys \
                            and 'QCLOUD' in variable_keys and 'HGT' in variable_keys and 'RAINC' in variable_keys \
                            and 'RAINNC' in variable_keys and 'SWDOWN' in variable_keys and 'GSW' in variable_keys \
                            and 'GLW' in variable_keys and 'HFX' in variable_keys and 'QFX' in variable_keys \
                            and 'LH' in variable_keys and 'TT' in variable_keys and 'GHT' in variable_keys \
                            and 'RH' in variable_keys and 'SLP' in variable_keys:
                        xlon = ds.variables['XLONG'][:][0][0].tolist()
                        if 28 < min(xlon) and max(xlon) < 132:
                            # print('add')
                            res.append([(ymd_h(spot_str[i]) + hours(12)).strftime(ymdh),
                                        (ymd_h(pred_str[i]) - hours(2 * 24)).strftime(ymdh),
                                        var['mesh_lat'],
                                        var['mesh_lon'],
                                        ds.variables['U10'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['V10'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['UU'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['VV'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['Q2'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['T2'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['PSFC'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['QVAPOR'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['QCLOUD'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['HGT'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['RAINC'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['RAINNC'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['SWDOWN'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['GSW'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['GLW'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['HFX'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['QFX'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['LH'][:, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['TT'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['GHT'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['RH'][:, 0, var['mesh_lat'], var['mesh_lon']],
                                        ds.variables['SLP'][:, var['mesh_lat'], var['mesh_lon']]
                                        ])
    return res


def get_wrf_time(wrf_dir):
    fileNames = os.listdir(wrf_dir)
    dirs = []
    for name in fileNames:
        if len(name) > 35:
            dirs.append(name.split('.')[-2].split('_')[0])
    return pd.to_datetime(dirs, format=ymdh)


def dumpWRF(near_mesh, wrf_dir, wrf_times, out_name, year):
    wrf_pred = []
    wrf_pred_full_path = wrf_pre_dir + '/' + out_name + '_pred_' + str(year) + '.pkl'
    wrf_pred = [dump_wrf_var(near_mesh, wrf_dir, wrf_times, wrf_times + hours(7 * 24 + 12), str(year))]
    with open(wrf_pred_full_path, 'wb') as of:
        pickle.dump(wrf_pred, of)


near_mesh = pd.read_csv('../../data/mesh/near_mesh.csv')
wrf_times = get_wrf_time(wrf_dir).sort_values()
dumpWRF(near_mesh, wrf_dir, wrf_times, 'wrf', 2013)
