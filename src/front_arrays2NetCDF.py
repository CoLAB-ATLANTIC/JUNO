
# SCRIPT to get the numpy arrays of fronts related to the Cayula Cornillon Algorithm (sotred in the MUR_daily_fronts_npy folder)
# and use them to create a netCDF file with all that data
# With this netCDF we are going to be able to easily calculate a frontal probabilites of the CCA for a given period

#Import Libraries

import numpy as np
import os
import netCDF4 as nc
import glob
import datetime


nc_file = os.getcwd()
nc_file = os.path.join('JUNO/data/CCA_MUR_fronts.nc')


ds = nc.Dataset(nc_file, 'w', format='NETCDF4')

ds.title = 'CCA MUR Fronts Arrays'

time = ds.createDimension('time')
lat = ds.createDimension('lat', 1001)
lon = ds.createDimension('lon', 1401)

times = ds.createVariable('time', 'f4', ('time', ))
lats = ds.createVariable('lat', 'f4', ('lat', ))
lons = ds.createVariable('lon', 'f4', ('lon', ))
value = ds.createVariable('value', 'u1', ('time', 'lat', 'lon',))
value.units = 'Unknown'
times.units = 'days since 1-1-1'

lats[:] = np.linspace(35, 45, 1001)
lons[:] = np.linspace(-19, -5, 1401)

base_path = os.getcwd()
base_path = os.path.join('JUNO')

count = 0    
dates = []

for filename in sorted(glob.glob((os.path.join(base_path, 'data/MUR_daily_fronts_npy/*.npy')))):
    
    date_str = filename.split('sst_', 1)[1]
    date_str = date_str.split('.', 1)[0]

    date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
    dates.append(date_obj)
    
    value[count, :, :] = np.load(filename)
    
    count += 1
    
#times = dates
dates_time = [tmp.toordinal() for tmp in dates]
print(dates_time)
#dates_time = date2num(dates, times.units)
times[:] = dates_time


ds.close()



