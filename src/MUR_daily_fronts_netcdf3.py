
################################################### MUR_daily_fronts_netcdf.py  #############################################
###                                                                                                                       ###
###    Through a cronjob MUR daily data is downloaded and stored in the MUR_daily_data folder.                            ###
###    Then the 3 algorithms are applied to this data to get the arrays of fronts (for the Canny, BOA and CCA)            ###
###    Then the 3 arrays plus the SST array are stored in a netCDF file int the MUR_daily_fronts_netcdf folder            ###
###                                                                                                                       ###
#############################################################################################################################



# Script which, through a daily cron job makes the donwload of daily MUR data and saves it in the MUR_daily_data folder.
# Then it applies the 3 algorithms to the data and stores the result arrays in a netCDF file


#Import Libraries
import pandas as pd
import numpy as np
import xarray as xr
import os
import cv2
import matplotlib
import geopandas
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter
from datetime import date, timedelta, datetime
#import wget
from math import floor
#from pydap.client import open_url
import netCDF4 as nc
import requests

matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import BOA
import CayulaCornillon_xarray


def get_base_dir():
    return os.environ.get("JUNO_BASE_DIR", "/app")


def get_data_dir():
    return os.environ.get("JUNO_DATA_DIR", os.path.join(get_base_dir(), "data"))


def get_atlantic_shapefile_path(data_dir):
    return os.environ.get(
        "JUNO_ATLANTIC_SHAPEFILE",
        os.path.join(data_dir, "atlantic_shapefile/aoi_atlantic_clip.shp"),
    )


def create_open_ocean_mask(data_xarray, shapefile_path):
    """
    Rasterize the open-ocean AOI shapefile onto the MUR lat/lon grid.
    True means Canny is allowed to detect fronts in that pixel.
    """

    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(
            "Atlantic AOI shapefile not found at "
            f"{shapefile_path}. Mount/copy the shapefile components "
            "(.shp, .shx, .dbf, .prj) into the data/atlantic_shapefile folder "
            "or set JUNO_ATLANTIC_SHAPEFILE to the correct .shp path."
        )

    ocean_area = geopandas.read_file(shapefile_path).to_crs("EPSG:4326")
    lons = data_xarray["lon"].values
    lats = data_xarray["lat"].values
    height = len(lats)
    width = len(lons)
    lon_res = abs(lons[1] - lons[0])
    lat_res = abs(lats[1] - lats[0])

    transform = from_bounds(
        float(lons.min() - lon_res / 2),
        float(lats.min() - lat_res / 2),
        float(lons.max() + lon_res / 2),
        float(lats.max() + lat_res / 2),
        width,
        height,
    )

    # geometry_mask is north-up; MUR arrays here are latitude-ascending.
    north_up_mask = geometry_mask(
        ocean_area.geometry,
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    return np.flipud(north_up_mask)




######################################### IMPORT DATA #######################################################

def get_data(nc_path):
    
    """
    Function to get our netCDF file that is stored in the MUR_daily_data folder
    and convert it to an xarray. The nc_name parameter is the string name of the netCDF file we want to import
    """
    
    base_path = os.getcwd()
    file_path = os.path.join(base_path, nc_path)  
    
    xarray_mur = xr.open_mfdataset(file_path, engine='netcdf4')
    
    #convert temperature from Kelvin to Celsius
    #xarray_mur['analysed_sst'] = xarray_mur['analysed_sst'] - 273.15
    
    # Rename dimensions
    xarray_mur = xarray_mur.rename({"latitude": "lat", "longitude": "lon"})
    
    return xarray_mur

#################################### CANNY ALGORITHM ##########################################################

def canny_front_detection_1day(
    data_xarray,
    open_ocean_mask=None,
    thresh_min=120,
    thresh_max=220,
    apertureSize=5,
    sigma=5,
):
    
    """
    This function receives a dataframe with MUR data for a individual day and returns the array 
    that result from the aplication of the Canny Algorithm from OpenCV. 
    For visualization purposes, one can change the minimum and maximum threshold.
    One can also apply a gaussian filter with a certain sigma value to reduce noise of the image.
    """
    
    #Get the sst array in the right shape
    sst = data_xarray['analysed_sst'][0,:,:].values.astype(float)
    valid_mask = np.isfinite(sst)
    if open_ocean_mask is not None:
        valid_mask = valid_mask & open_ocean_mask

    if not np.any(valid_mask):
        raise ValueError("No valid open-ocean SST pixels available for Canny front detection.")

    #sst = np.array(data_xarray['analysed_sst'])
    #sst = np.squeeze(sst)
    #Convert Temperature values to uint8 format with values in the range of 0-255
    sst_min = np.nanmin(sst[valid_mask])
    sst_max = np.nanmax(sst[valid_mask])
    sst_range = sst_max - sst_min
    if sst_range == 0:
        return np.full_like(sst, np.nan, dtype=float)

    sst_final = ((sst - sst_min) * (1 / sst_range * 255))
    sst_final[~valid_mask] = np.nan

    if not np.all(valid_mask):
        nearest_valid_indexes = distance_transform_edt(
            ~valid_mask,
            return_distances=False,
            return_indices=True,
        )
        nearest_valid_values = sst_final[tuple(nearest_valid_indexes)]
        sst_final[~valid_mask] = nearest_valid_values[~valid_mask]

    sst_final = sst_final.astype('uint8')
    sst_final = np.flipud(sst_final)   #flipud -> Reverse the order of elements along axis 0 (up/down).
    canny_mask = np.flipud(valid_mask)
    #in case we want to apply a gaussian filter with a certain sigma value (by default is 0)
    sst_final = gaussian_filter(sst_final, sigma=sigma)   

    
    #apply the canny algorithm and plot the image with the edges
    canny = cv2.Canny(sst_final, thresh_min, thresh_max, apertureSize=apertureSize, L2gradient=False)
    canny[~binary_erosion(canny_mask, structure=np.ones((3, 3)))] = 0
    
    canny[canny == 255] = 1
    
    canny = canny.astype('float')
    canny[canny == 0] = np.nan
    
    canny_front = np.flipud(canny)
    canny_front[~valid_mask] = np.nan
    
    
    return canny_front
    



########################################################################################################################
################################### Belkin O'Reilly Algorithm ##########################################################



def BOA_aplication(data_xarray, threshold = 0.05):  
    
    """
    Function to, for a given dataframe with a longitude, latitude and SST columns, 
    identifies fronts through the application of BOA algorithm.
    We also need to define a threshold value to later get the frontal probabilities matrix
    (if the pixel value is greater than the threshold, then it is considered a front, otherwise don't). 
    """
    
    lon = np.array(data_xarray['lon']).astype('float64')
    lat = np.array(data_xarray['lat']).astype('float64')
    
    ingrid = data_xarray['analysed_sst'][0,:,:].values
   # ingrid = np.array(data_xarray['analysed_sst'])
    #ingrid = np.squeeze(ingrid)
    
    
    boa_front = BOA.boa(lon=lon, lat=lat, ingrid=ingrid, nodata = np.nan, direction = False)
    boa_front = np.flip(boa_front, axis=0)
    boa_front = np.array([[boa_front[j][i] for j in range(len(boa_front))] for i in range(len(boa_front[0])-1,-1,-1)])
    
    boa_front = np.where(boa_front>=threshold, 1, boa_front)    
    boa_front = np.where(boa_front<threshold, 0, boa_front)
    
    #convert 0s to Nans
    boa_front[boa_front == 0] = 'nan'

    boa_front = np.flipud(boa_front) 

    return boa_front



################################## CAYULA-CORNILLON ALGORITHM #################################################################
    

    
def CCA_front(data_xarray): 
    
    """
    Function that calculates the fronts matrix. Given an image (SST data respective to one day) it applies the
    Cayula-Cornillon Algorithm for Single Image Edge Detection (CCA-SIED) to discover the fronts.
    It returns the matrix with the fronts: if pixel = 1 it was considered a front, otherwise, pixel = 0
    It basically converts the (x,y) coordinate points to indexes of the frontal probability matrix. These indexes are considered fronts
    The df parameter is the dataframe with the SST data for a certain day
    """
    
    front = np.zeros((1001,1401))       #initialize a matrix of zeros. This shape is for the MUR data
    
    #2 empty arrays that will store the x and y values of the lines that are suposed to be drawn
    x = np.array([])
    y = np.array([])
        
    xdata_final, ydata_final = CayulaCornillon_xarray.CCA_SIED(data_xarray)       
    x = np.append(x, xdata_final)
    y = np.append(y, ydata_final)
        
    cols_x = np.array([])
    for value in x:                     #convert values in array x to the respective index in a (1001, 1401) matrix
        aux_x = (19+value)/0.01                  #these numbers are relative to the MUR data
        cols_x = np.append(cols_x, aux_x)
    
    rows_y = np.array([])
    for value in y:                     #convert values in array y to the respective index in a (1001, 1401) matrix
        aux_y = (45-value)/0.01                  #these numbers are relative to the MUR data
        rows_y = np.append(rows_y, aux_y)
     
    cols_x = np.round(cols_x)
    rows_y = np.round(rows_y)
            
    for i in range(len(cols_x)):   #it could also be len(rows_y)
        front[int(rows_y[i]), int(cols_x[i])] = front[int(rows_y[i]), int(cols_x[i])] + 1
            
    front[front != 0] = 1
    
    #convert 0s to Nans
    front[front == 0] = 'nan'

    cca_front = np.flipud(front) 
    
    return cca_front
    
    ##########################################################################################################################################
    
    ################################# GET THE REAL SST IMAGE FROM THE CMEMS FORECAST DATASET ##############################################
    
    
def real_sst_image(data_xarray):
        
    """
    Function to store the real sst image
    """
    
    sst_image = data_xarray['analysed_sst'][0,:,:].values

    return sst_image
 


def main():
    
    data_dir = get_data_dir()
    
    #base_path = os.path.join(base_path, '../')      #local machine
    
    
    #download MUR data for the day before yesterday
    day_txt = (date.today() - timedelta(days=2)).strftime('%Y%m%d')
    
    
    date_obj = datetime.strptime(day_txt, "%Y%m%d")
    day_txt_f = datetime.strftime(date_obj, "%Y-%m-%d")
    
        
    exist_path = os.path.exists(os.path.join(data_dir, 'MUR_daily_data'))   #check if folder MUR_dailyu_data exists in data folder
    if not exist_path:                                                            #if it don't exist:
        os.makedirs(os.path.join(data_dir, 'MUR_daily_data'))               #create the folder

    #check if the daily sst data file already exists in the MUR_daily_data folder. If it does delete it  
    exist_sst_file = os.path.join(data_dir, 'MUR_daily_data/sst_' + day_txt + '.nc')
    if os.path.exists(exist_sst_file):
        os.remove(exist_sst_file)
        
        
        
##################################################### DOWNLOAD MUR DATA ############################################################################
    
    url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.nc?analysed_sst[(" + day_txt_f + "T09:00:00Z):1:(" + day_txt_f + "T09:00:00Z)][(35):1:(45)][(-19):1:(-5)]"
    
    output_file = os.path.join(data_dir, "MUR_daily_data/sst_" + day_txt + '.nc')
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"NetCDF file saved as {output_file}")
    else:
        print("Failed to download the NetCDF file.")
        
    
    nc_path = os.path.join(output_file)
    
    xarray_mur = get_data(nc_path)
    open_ocean_mask = create_open_ocean_mask(xarray_mur, get_atlantic_shapefile_path(data_dir))
    
    
    sst_image = real_sst_image(xarray_mur)
    
    canny_front = canny_front_detection_1day(xarray_mur, open_ocean_mask=open_ocean_mask)
    
    boa_front = BOA_aplication(xarray_mur, threshold=0.05)
    
    cca_front = CCA_front(xarray_mur)
        

    exist_path = os.path.exists(os.path.join(data_dir, 'MUR_daily_fronts_netcdf'))    #check if folder MUR_algorithm_daily_images exists in data folder
    if not exist_path:                                                                         #if doesn't exist
        os.makedirs(os.path.join(data_dir, 'MUR_daily_fronts_netcdf'))                # create the folder
        
        
        
    ################################################### CREATION OF THE NETCDF   #######################################################

    nc_file = os.path.join(data_dir, 'MUR_daily_fronts_netcdf', day_txt + '00.nc')
    
    #nc_file = os.path.join(nc_file, '../data/MUR_daily_fronts_netcdf/MUR' + day_txt + '.nc')    #LOCAL MACHINE
    
    
    
    if os.path.exists(nc_file):
        os.remove(nc_file)

    ds = nc.Dataset(nc_file, 'w', format='NETCDF4')

    ds.title = 'MUR ' + day_txt + ' Fronts Arrays (Xarrays)'

    #create dimensions of the NetCDF file
    #time = ds.createDimension('time')
    lat = ds.createDimension('lat', 1001)
    lon = ds.createDimension('lon', 1401)

    #times = ds.createVariable('time', 'f4', ('time', ))
    lats = ds.createVariable('lat', 'f4', ('lat', ))
    lons = ds.createVariable('lon', 'f4', ('lon', ))

    sst_analyzed = ds.createVariable('sst', 'f4', ('lat', 'lon',))    #('lat', 'lon',)
    sst_analyzed.units = 'C'   #degrees Celsius
    sst_analyzed.description = 'Array with the Sea-Surface Temperature (SST) in ºC relative to the MUR data for that day'
    #sst_analyzed[0, :, :] = sst
    sst_analyzed[:, :] = sst_image


    canny = ds.createVariable('Canny', 'f4', ('lat', 'lon',))
    canny.units = 'Unknown'
    canny.description = 'Array with identyfied fronts through Canny from OpenCV (1-> front), (Nan->not front)'
    #canny[0, :, :] = canny_front.astype(float)
    canny[:, :] = canny_front.astype(float)
    
    boa = ds.createVariable('BOA', 'f4', ('lat', 'lon',))
    boa.units = 'Unknown'
    boa.description = 'Array with identyfied fronts through the Belkin O Reilly Algorithm (temperature gradient). If the gradient is bigger than certain threshold is considered front (1) otherwise Nan'
    #boa[0, :, :] = boa_front
    boa[:, :] = boa_front
    
    cca = ds.createVariable('CCA', 'f4', ('lat', 'lon',))
    cca.units = 'Unknown'
    cca.description = 'Array with identyfied fronts through the Cayula Cornillon Algorithm (1->front) (Nan->not front)'
    #cca[0, :, :] = cca_front.astype(float)
    cca[:, :] = cca_front.astype(float)
    
    #times.units = 'days since 1-1-1'

    lats[:] = np.linspace(35, 45, 1001)
    lons[:] = np.linspace(-19, -5, 1401)


# date_obj = datetime.datetime.strptime(day_txt, '%Y%m%d')
    #date_time = date_obj.toordinal()
    #times[:] = date_time

    ds.close()
    

if __name__ == "__main__":
    main()
