import pandas as pd

wrf_gts_path = '../../data/output'
era_path = '../../data/era_pre'

for year in range(2013, 2018, 1):
    wrf_gts = pd.read_csv(wrf_gts_path + '/dataset_' + str(year) + '.csv', index_col=0)
    print(wrf_gts.shape[0])
    era = pd.read_csv(era_path + '/era_' + str(year) + '.csv', index_col=0)
    print(era.shape[0])
    res = wrf_gts.merge(era, left_on=['Station_Id', 'PredTime'], right_on=['Station_Id', 'Time'])
    res.rename(columns={'Direction_x': 'Direction_wrf',
                        'Speed_x': 'Speed_wrf',
                        'Direction_y': 'Direction_gts',
                        'Speed_y': 'Speed_gts',
                        'Direction': 'Direction_era',
                        'Speed': 'Speed_era'}, inplace=True)
    res.to_csv('../../data/output/wrf_gts_era_' + str(year) + '.csv', index=False, encoding='utf-8')
    print(res.columns.values)
