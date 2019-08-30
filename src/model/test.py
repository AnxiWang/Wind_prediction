import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type':'reanalysis',
        'format':'netcdf',
        'year':'2013',
        'month':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12'
        ],
        'day':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'time':[
            '00:00','06:00','12:00',
            '18:00'
        ],
        'pressure_level':[
            '100','200','500',
            '700','850','1000'
        ],
        'variable':[
            'geopotential','specific_humidity','temperature',
            'u_component_of_wind','v_component_of_wind'
        ]
    },
    '2013.nc')