
################################################### CMEMS_daily_fronts_netcdf.py  #############################################################
###                                                                                                                                         ###
###    In this script the CMEMS daily data (2 days behind) is downloaded and stored in the CMEMS_daily_data folder.                 ###
###    Then the 3 algorithms are applied to this data to get the arrays of fronts (for the Canny, BOA and CCA)                              ###
###    Then the 3 arrays plus the SST array are stored in a netCDF file int the CMEMS_daily_fronts_netcdf folder                            ###
###                                                                                                                                         ###
###############################################################################################################################################


from datetime import date, timedelta
import numpy as np
import xarray as xr
import os
from scipy.ndimage import gaussian_filter
import matplotlib
import cv2
import netCDF4 as nc
import datetime
import copernicusmarine
import geopandas
import rioxarray as rio
from shapely.geometry import mapping


matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import BOA
import CayulaCornillon_xarray    #CayulaCornillon after making some profiling changes to improve efficiency



######################################### IMPORT DATA ################################################################

def get_data(data, base_path):
    
    """
    Function to get our netCDF file that is stored in the data directory inside the MUR_seasonal_data folder and convert it to a dataframe.
    The data parameter is the string name of the netCDF file we want to import
    """
    
    #base_path = base_path
    data_folder = os.path.join(base_path, "data/CMEMS_daily_data")  
    
    nc_path = os.path.join(data_folder, data)
    data_xarray = xr.load_dataset(nc_path)
    
    #para estarem todos com a mesma nomenclatura usada o CCA_SIED do script CayulaCornillon_xarray.py
    data_xarray = data_xarray.rename({'latitude':'lat', 'longitude':'lon', 'thetao':'analysed_sst'})
    
    # Select the first level of the 'depth' dimension (assuming depth size is 1)
    data_xarray = data_xarray.isel(depth=0)
    
    # Since we've selected the only 'depth' level, 'depth' becomes an unnecessary coordinate and can be dropped
    data_xarray = data_xarray.drop_vars('depth')
    
    return data_xarray


#################################### CANNY ALGORITHM ##########################################################

def canny_application(data_xarray, thresh_min=100, thresh_max=150, apertureSize=5, sigma=3):
    
    """
    This function receives a dataframe with CMEMS Forecast data for a individual day and returns the array 
    that result from the aplication of the Canny Algorithm from OpenCV. 
    For visualization purposes, one can change the minimum and maximum threshold.
    One can also apply a gaussian filter with a certain sigma value to reduce noise of the image.
    """
    
    #Get the sst array in the right shape
    sst = np.array(data_xarray['analysed_sst'])
    sst = np.squeeze(sst)
    #Convert Temperature values to uint8 format with values in the range of 0-255
    sst_final = ((sst - np.nanmin(sst)) * (1/(np.nanmax(sst) - np.nanmin(sst)) * 255)).astype('uint8')
    sst_final = np.flipud(sst_final)   #flipud -> Reverse the order of elements along axis 0 (up/down).
    #in case we want to apply a gaussian filter with a certain sigma value (by default is 0)
    sst_final = gaussian_filter(sst_final, sigma=sigma)   

    #apply the canny algorithm and plot the image with the edges
    canny = cv2.Canny(sst_final, thresh_min, thresh_max, apertureSize=apertureSize, L2gradient=False)
    
    canny[canny == 255] = 1
    
    #convert 0s to Nans
    canny = canny.astype('float')
    canny[canny == 0] = 'nan'
    
    #Apply a mask for the continental zone:
    mask = np.isnan(np.flipud(sst))    #Boolean array: True where array Temp had Null Values (correspond to the continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which values= 255 when mask=True
    #Dilation to ensure that the pixels that belong to the "shore/continental zone" are not considered fronts 
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    canny_front = np.ma.masked_array(canny, mask_dilated)   #Mask an array where a condition is True
    
    canny_front = np.flipud(canny_front)    
    
    
    return canny_front


########################################################################################################################

################################### Belkin O'Reilly Algorithm ##########################################################


def BOA_aplication(data_xarray, threshold = 0.4):  
    
    """
    Function to, for a given dataframe with a longitude, latitude and SST columns, 
    identifies fronts through the application of BOA algorithm.
    We also need to define a threshold value to later get the frontal probabilities matrix
    (if the pixel value is greater than the threshold, then it is considered a front, otherwise don't). 
    """
    
    lon = np.array(data_xarray['lon']).astype('float64')
    lat = np.array(data_xarray['lat']).astype('float64')
    ingrid = np.array(data_xarray['analysed_sst'])
    ingrid = np.squeeze(ingrid)
    
    
    boa_front = BOA.boa(lon=lon, lat=lat, ingrid=ingrid, nodata = np.nan, direction = False)
    boa_front = np.flip(boa_front, axis=0)
    boa_front = np.array([[boa_front[j][i] for j in range(len(boa_front))] for i in range(len(boa_front[0])-1,-1,-1)])
    
    boa_front = np.where(boa_front>=threshold, 1, boa_front)    
    boa_front = np.where(boa_front<threshold, 0, boa_front)
    
    #convert 0s to Nans
    boa_front[boa_front == 0] = 'nan'


    mask = np.isnan(np.flipud(ingrid))       #Boolean array=True where array Temp had Null values (continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    boa_front = np.ma.masked_array(boa_front, mask_dilated)  
    
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
    
    #front = np.zeros((361, 505))       #initialize a matrix of zeros. This shape is for the MUR data
    #initialize a matrix of zeros with the shape (coordinates) 0f the data (lat X lon)
    front = np.zeros((len(data_xarray.lat), len(data_xarray.lon)))
    
    #2 empty arrays that will store the x and y values of the lines that are suposed to be drawn
    x = np.array([])
    y = np.array([])
        
    xdata_final, ydata_final = CayulaCornillon_xarray.CCA_SIED(data_xarray)       
    x = np.append(x, xdata_final)
    y = np.append(y, ydata_final)
        
    cols_x = np.array([])
    for value in x:                     #convert values in array x to the respective index in a (1001, 1401) matrix
        #aux_x = (19+value)/0.027723                  #these numbers are relative to the MUR data
        aux_x = (abs(min(data_xarray.lon.values))+value)/(data_xarray.lon.values[1] - data_xarray.lon.values[0])
        cols_x = np.append(cols_x, aux_x)
    
    rows_y = np.array([])
    for value in y:                     #convert values in array y to the respective index in a (1001, 1401) matrix
        #aux_y = (45-value)/0.0277                  #these numbers are relative to the MUR data
        aux_y = (max(data_xarray.lat.values)-value)/(data_xarray.lon.values[1] - data_xarray.lon.values[0])
        rows_y = np.append(rows_y, aux_y)
     
    cols_x = np.round(cols_x)
    rows_y = np.round(rows_y)
            
    for i in range(len(cols_x)):   #it could also be len(rows_y)
        front[int(rows_y[i]), int(cols_x[i])] = front[int(rows_y[i]), int(cols_x[i])] + 1
            
    front[front != 0] = 1
    
    #convert 0s to Nans
    front[front == 0] = 'nan'


    #Create a masked_array in order to get the continental zone well defined
    
    #Convert some df to a numpy array with the SST values for each value of longitude and latitude
    sst = np.array(data_xarray['analysed_sst'])
    sst = np.squeeze(sst)
    mask = np.isnan(np.flipud(sst))       #Boolean array=True where array Temp had Null values (continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    cca_front = np.ma.masked_array(front, mask_dilated)  
    
    cca_front = np.flipud(cca_front) 
    
    return cca_front

  
##########################################################################################################################################
    
 ################################# GET THE REAL SST IMAGE FROM THE CMEMS FORECAST DATASET ##############################################
    
    
def real_sst_image(data_xarray):
        
    """
    Function to store the real sst image
    """
    
    sst = np.array(data_xarray['analysed_sst'])
    sst = np.squeeze(sst)
    
    return sst
    
    
######################################################   MAIN   ################################################################################

def main():
    
    base_path = os.getcwd()
    base_path = '/home/colabatlantic2/projects/JUNO/'
    #base_path = os.path.join(base_path, 'projects/JUNO')       #servidor
    #base_path = '/home/luisfigueiredo/edgeDetection/'
    
    #My Username and Password are stored in a .txt file stored in a data folder which belong to the gitignore
    with open('/home/colabatlantic2/projects/JUNO/data/copernicus_login.txt') as f:
        #with open('../data/copernicus_login.txt') as f:
        lines = f.readlines()
        
    USERNAME = lines[0][:-1]    #SERVIDOR
    PASSWORD = lines[1][:-1]
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/CMEMS_daily_data'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/CMEMS_daily_data'))
        

    #Get the data in the format we want
    iday = 7
    filename_day_txt = (date.today() + timedelta(days=iday)).strftime('%Y-%m-%d')
    day_txt = (date.today() + timedelta(days=iday)).strftime('%Y-%m-%d')
    #date_txt = day_txt + ' 00:00:00'
    date_txt = day_txt + 'T00:00:00'
    
    #check if the daily sst data file already exists in the CMEMS_daily_data folder. If it does delete it  
    exist_sst_file = os.path.join(base_path, 'data/CMEMS_daily_data/CMEMS_' + filename_day_txt + '.nc')
    if os.path.exists(exist_sst_file):
        os.remove(exist_sst_file)
        

    OUTPUT_FILENAME = 'CMEMS_' + day_txt +'.nc'
    OUTPUT_DIRECTORY = '/home/colabatlantic2/projects/JUNO/data/CMEMS_daily_data'
    #OUTPUT_DIRECTORY = '/home/luisfigueiredo/edgeDetection/data/CMEMS_daily_data'
    
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        # dataset_version="202211",
        variables=["thetao"],
        minimum_longitude=-100.04166666666666,
        maximum_longitude=50.04166666666667,
        minimum_latitude=-70.04166666666666,
        maximum_latitude=80.04166666666667,
        minimum_depth=0,
        maximum_depth=0,
        username=USERNAME,
        password=PASSWORD,
        start_datetime=date_txt,
        end_datetime=date_txt,
        output_filename = OUTPUT_FILENAME,
        output_directory = OUTPUT_DIRECTORY,
        force_download = True
        )
    
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/CMEMS_daily_fronts_netcdf'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/CMEMS_daily_fronts_netcdf'))
  
    xarray_cmems = get_data('CMEMS_' + day_txt + '.nc', base_path=base_path)   
    
    canny_front = canny_application(xarray_cmems)
    
    boa_front = BOA_aplication(xarray_cmems, threshold=0.6)
    
    cca_front = CCA_front(xarray_cmems)
        
    sst = real_sst_image(xarray_cmems)
    
    
    ################################################### CREATION OF THE NETCDF   #######################################################
    
    #nc_file = os.getcwd()
    nc_file = "/home/colabatlantic2/"
    #nc_file = '/home/luisfigueiredo/edgeDetection'
    nc_file = os.path.join(nc_file, 'projects/JUNO/data/CMEMS_daily_fronts_netcdf/' + day_txt.replace("-","") + '00.nc')
    #nc_file = os.path.join(nc_file, 'data/CMEMS_daily_fronts_netcdf/CMEMS' + day_txt + '.nc')

    ds = nc.Dataset(nc_file, 'w', format='NETCDF4')

    ds.title = 'CMEMS_ ' + day_txt + ' Fronts Arrays (Xarrays)'

    #create dimensions of the NetCDF file
    #time = ds.createDimension('time')
    nr_lat = len(xarray_cmems.lat)
    nr_lon = len(xarray_cmems.lon)
    lat = ds.createDimension('lat', nr_lat)
    lon = ds.createDimension('lon', nr_lon)

    #times = ds.createVariable('time', 'f4', ('time', ))
    lats = ds.createVariable('lat', 'f4', ('lat', ))
    lons = ds.createVariable('lon', 'f4', ('lon', ))

    sst_analyzed = ds.createVariable('sst', 'f4', ('lat', 'lon',))     #('time', 'lat', 'lon',)
    sst_analyzed.units = 'C'   #degrees Celsius
    sst_analyzed.description = 'Array with the Sea-Surface Temperature (SST) in ºC relative to the CMEMS data for that day'
    #sst_analyzed[0, :, :] = sst
    sst_analyzed[:, :] = sst

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

    max_lon = max(xarray_cmems.lon.values)
    min_lon = min(xarray_cmems.lon.values)
    max_lat = max(xarray_cmems.lat.values)
    min_lat = min(xarray_cmems.lat.values)
    
    lats[:] = np.linspace(min_lat, max_lat, nr_lat)
    lons[:] = np.linspace(min_lon, max_lon, nr_lon)
   
   
    #date_obj = datetime.datetime.strptime(day_txt, '%Y-%m-%d')
    #date_time = date_obj.toordinal()
    #times[:] = date_time

    ds.close()
    
    #nc_file = os.path.join(nc_file, 'projects/JUNO/data/CMEMS_daily_fronts_netcdf/' + day_txt.replace("-","") + '00.nc')
    
    #shape_path = os.getcwd()
    shape_path = '/home/colabatlantic2/'
    
    SHPFILE_PATH = os.path.join(shape_path, 'projects/JUNO/data/atlantic_shapefile/aoi_atlantic_clip.shp' )
 
    sf = geopandas.read_file(SHPFILE_PATH)
    #sf.set_crs('epsg:4326', inplace = True, allow_override = True)
 
    netcdf_file = xr.open_dataset(nc_file, engine='netcdf4')
    netcdf_file.rio.write_crs('epsg:4326', inplace = True)
    netcdf_file = netcdf_file.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    clipped_nc = netcdf_file.rio.clip(sf.geometry.apply(mapping), sf.crs, all_touched = True)
    
    
    variable_names = ["BOA", "Canny", "CCA"]

    for var_name in variable_names:
        clipped_nc[var_name] = clipped_nc[var_name].where(clipped_nc[var_name] <= 1, np.nan)

    
    # Close the original dataset to free up the file
    netcdf_file.close()
    clipped_nc.to_netcdf(nc_file)
    

if __name__ == "__main__":
    main()

