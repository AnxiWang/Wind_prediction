import pandas as pd

wrf_gts_path = '../../data/output'
era_path = '../../data/era_pre'

for year in range(2013, 2018, 1):
    wrf_gts = pd.read_csv(wrf_gts_path + '/dataset_' + str(year) + '.csv', index_col=0)
    print(wrf_gts.shape[0])
    era = pd.read_csv(era_path + '/era_' + str(year) + '.csv', index_col=0)
    print(era.shape[0])
    res = wrf_gts.merge(era, left_on=['Station_Id', 'PredTime'], right_on=['Station_Id', 'Time'])
    res.to_csv('../../data/output/wrf_gts_era_' + str(year) + '.csv', index=False, encoding='utf-8')
