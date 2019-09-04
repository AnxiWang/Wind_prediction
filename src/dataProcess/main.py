from wrfDataProcess import *
from windFeature import *

GTSPath = '/home/shared_data/Wind_WRF/Data1/GTS_OUT'

if __name__ == "__main__":
    for i in range(2013, 2020, 1):
        # dataSetPath = GTSPath + '/' + str(i)
        # readGTS(dataSetPath, str(i))
        if i < 2018:
            GTS = gatherGTS(i)
            wrf_pred = dumpWRF(i)
            WRF = meshWRFtoStation(wrf_pred, i)
            dataset = buildDataset(i)
        if i >= 2018:
            pass