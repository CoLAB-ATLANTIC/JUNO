import numpy as np
import os
import netCDF4 as nc
import xarray as xr
import datetime
import itertools
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

base_path = os.getcwd()
os.path.join(base_path, 'JUNO')

nc_path = os.path.join(base_path, 'data/CCA_MUR_fronts.nc')
ds = nc.Dataset(nc_path)
data_xarray = xr.load_dataset(nc_path, decode_times=False)     #data is an xarray

#get a list of all the dates present in the xarray
list_dates = []
for i in range(len(data_xarray['time'])):
    list_dates.append(datetime.datetime.fromordinal(int(data_xarray['time'][i])))


#get a list with datetime.datetime de todos os dias dos Verões (Jul, Ago, Set) de 2012 a 2021
summer_10years = []
start_year = 2012
while start_year < 2022:
    summer_10years.append(pd.date_range(start=str(start_year)+'-06-01', end = str(start_year)+'-08-31').to_pydatetime().tolist())
    start_year += 1
summer_10years = list(itertools.chain.from_iterable(summer_10years))   #convert list of list to only 1 list


#get the frontal probabilities arrays for the dates we want (in this case summer of the last 10 years())
count = 0
fronts = np.zeros((1001, 1401))
for i in range(len(data_xarray.time)):
    if list_dates[i] in summer_10years:
        fronts = fronts + data_xarray.value[i]
        count += 1
frontal_prob = fronts/count

#if folder MUR_seasonal_images does not exist, creat it
exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_seasonal_images'))
if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/MUR_seasonal_images'))


################################################## Get MUR data for 1 day to Map the Continental Zone  ###########################################
nc_map_path = os.path.join(base_path, "data/MUR_single_days/sst_20190615.nc")     # Minha maquina é 2022 (não 2019)
ds = nc.Dataset(nc_map_path)
data_map = xr.load_dataset(nc_map_path)

#Convert the netCDF file to a dataframe
datadf = data_map.to_dataframe()
datadf = datadf.reset_index()
datadf['analysed_sst'] =  datadf['analysed_sst']-273.15    #convert temperature to celsius

sst = datadf.pivot_table(index='lon', columns='lat', values='analysed_sst').T.values
mask = np.isnan(np.flipud(sst))    #Boolean array: True where array Temp had Null Values (correspond to the continental zone)
mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which values= 255 when mask=True
#Dilation to ensure that the pixels that belong to the "shore/continental zone" are not considered fronts 
kernel = np.ones((3,3), np.uint8)
mask_dilated = cv2.dilate(mask255, kernel)
frontal_prob =np.ma.masked_where(mask_dilated==255, frontal_prob)   #Mask an array where a condition is met.


################################## Frontal Probabilities Map with Continental Zone Mapped #######################################################
lat = data_xarray['lat'].values
lon = data_xarray['lon'].values
#Visualization purposes: continenal area in gray, and pixels with value=0 in white   
viridis = matplotlib.cm.get_cmap('viridis', 30)
newcolor = viridis(np.linspace(0,1,30))
white = np.array([1, 1, 1, 1])
newcolor[0, :] = white
newcmp = ListedColormap(newcolor)
newcmp.set_bad(color='gray')

plt.figure()
plt.imshow(frontal_prob, cmap=newcmp, extent = [lon[0], lon[-1], lat[0], lat[-1]], interpolation='bilinear')    
#extent is to define the extention of the x and y axis
plt.title("Cayula-Cornillon Algorithm Frontal Probability (MUR) ", fontsize=20)
plt.colorbar(orientation='horizontal', fraction=0.025, pad=0.08, aspect=50)
plt.savefig(os.path.join(base_path, 'data/MUR_seasonal_images/CCA_frontal_prob_Summer_2012To2021.jpg'))