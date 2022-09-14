
# Script that through a cron job downloads the daily data from the CMEMS Forecast (with 2 days delay but that can be changed), 
# and saves them as a .nc file in the data folder: CMEMS_forecast_daily_data. 
# Then it fetches this data and applies the 3 algorithms to it. (The data is used in xarray format). 
# The result is save in a daily NetCDF file


import motuclient
from datetime import date, timedelta
import numpy as np
import xarray as xr
import os
from scipy.ndimage import gaussian_filter
import matplotlib
import cv2
import netCDF4 as nc
import datetime

matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import BOA
import CayulaCornillon_xarray    #CayulaCornillon after making some profiling changes to improve efficiency


################################################## DOWNLOAD CMEMS_REANALYSIS DATA #####################################################

# this class will be used to parse the motuclient options from a dictionary:
class MotuOptions:
    def __init__(self, attrs: dict):
        super(MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None
        
#This objective of this function is:   
# post-process the script_template (displayed clicking on VIEW SCRIPT) to create a dictionary; returns this dictionary to feed the download of the data request
def motu_option_parser(script_template, usr, pwd, output_filename, output_directory):
    dictionary = dict([e.strip().partition(" ")[::2] for e in script_template.split('--')])
    dictionary['variable'] = [value for (var, value) in [e.strip().partition(" ")[::2] for e in script_template.split('--')] if var == 'variable']  
    for k, v in list(dictionary.items()):
        if v == '<OUTPUT_DIRECTORY>':
            dictionary[k] = output_directory
        if v == '<OUTPUT_FILENAME>':
            dictionary[k] = output_filename
        if v == '<USERNAME>':
            dictionary[k] = usr
        if v == '<PASSWORD>':
            dictionary[k] = pwd
        if k in ['longitude-min', 'longitude-max', 'latitude-min', 'latitude-max']:
            dictionary[k] = float(v)
        if k in ['date-min', 'date-max']:
            dictionary[k] = v[1:-1]
        dictionary[k.replace('-','_')] = dictionary.pop(k)
    dictionary.pop('python')
    dictionary['auth_mode'] = 'cas'
    return dictionary


####################################################################################################################

######################################### IMPORT DATA ################################################################

def get_data(data, base_path):
    
    """
    Function to get our netCDF file that is stored in the data directory inside the MUR_seasonal_data folder and convert it to a dataframe.
    The data parameter is the string name of the netCDF file we want to import
    """
    
    base_path = base_path
    data_folder = os.path.join(base_path, "data/MUR_daily_data")  
    
    nc_path = os.path.join(data_folder, data)
    data_xarray = xr.load_dataset(nc_path)
    
    return data_xarray


#################################### CANNY ALGORITHM ##########################################################

def canny_front_detection_1day(data_xarray, thresh_min=120, thresh_max=220, apertureSize=5, sigma=5):
    
    """
    This function receives a dataframe with MUR data for a individual day and returns the array 
    that result from the aplication of the Canny Algorithm from OpenCV. 
    For visualization purposes, one can change the minimum and maximum threshold.
    One can also apply a gaussian filter with a certain sigma value to reduce noise of the image.
    """
    
    #Get the sst array in the right shape
    sst = np.array(data_xarray['thetao'])
    sst = np.squeeze(sst)
    #Convert Temperature values to uint8 format with values in the range of 0-255
    sst_final = ((sst - np.nanmin(sst)) * (1/(np.nanmax(sst) - np.nanmin(sst)) * 255)).astype('uint8')
    sst_final = np.flipud(sst_final)   #flipud -> Reverse the order of elements along axis 0 (up/down).
    #in case we want to apply a gaussian filter with a certain sigma value (by default is 0)
    sst_final = gaussian_filter(sst_final, sigma=sigma)   

    #apply the canny algorithm and plot the image with the edges
    canny = cv2.Canny(sst_final, thresh_min, thresh_max, apertureSize=apertureSize, L2gradient=False)
    
    canny[canny == 255] = 1
    
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


def BOA_aplication(data_xarray, threshold = 0.05):  
    
    """
    Function to, for a given dataframe with a longitude, latitude and SST columns, 
    identifies fronts through the application of BOA algorithm.
    We also need to define a threshold value to later get the frontal probabilities matrix
    (if the pixel value is greater than the threshold, then it is considered a front, otherwise don't). 
    """
    
    lon = np.array(data_xarray['longitude']).astype('float64')
    lat = np.array(data_xarray['latitude']).astype('float64')
    ingrid = np.array(data_xarray['thetao'])
    ingrid = np.squeeze(ingrid)
    
    
    boa_front = BOA.boa(lon=lon, lat=lat, ingrid=ingrid, nodata = np.nan, direction = False)
    boa_front = np.flip(boa_front, axis=0)
    boa_front = np.array([[boa_front[j][i] for j in range(len(boa_front))] for i in range(len(boa_front[0])-1,-1,-1)])
    
    boa_front = np.where(boa_front>=threshold, 1, boa_front)    
    boa_front = np.where(boa_front<threshold, 0, boa_front)


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
    
    front = np.zeros((361, 505))       #initialize a matrix of zeros. This shape is for the MUR data
    
    #2 empty arrays that will store the x and y values of the lines that are suposed to be drawn
    x = np.array([])
    y = np.array([])
        
    xdata_final, ydata_final = CayulaCornillon_xarray.CCA_SIED(data_xarray)       
    x = np.append(x, xdata_final)
    y = np.append(y, ydata_final)
        
    cols_x = np.array([])
    for value in x:                     #convert values in array x to the respective index in a (1001, 1401) matrix
        aux_x = (19+value)/0.027723                  #these numbers are relative to the MUR data
        cols_x = np.append(cols_x, aux_x)
    
    rows_y = np.array([])
    for value in y:                     #convert values in array y to the respective index in a (1001, 1401) matrix
        aux_y = (45-value)/0.0277                  #these numbers are relative to the MUR data
        rows_y = np.append(rows_y, aux_y)
     
    cols_x = np.round(cols_x)
    rows_y = np.round(rows_y)
            
    for i in range(len(cols_x)):   #it could also be len(rows_y)
        front[int(rows_y[i]), int(cols_x[i])] = front[int(rows_y[i]), int(cols_x[i])] + 1
            
    front[front != 0] = 1


    #Create a masked_array in order to get the continental zone well defined
    
    #Convert some df to a numpy array with the SST values for each value of longitude and latitude
    sst = np.array(data_xarray['thetao'])
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
    
    sst = np.array(data_xarray['thetao'])
    sst = np.squeeze(sst)
    
    return sst
    
    

def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'projects/JUNO')       #servidor
    #base_path = os.path.join(base_path, 'JUNO')                #minha maquina
    
    #My Username and Password are stored in a .txt file stored in a data folder which belong to the gitignore
    with open('projects/JUNO/data/copernicus_login.txt') as f:   #quando fizer clone para o servidor esta documento .txt vai ser ignorado
        lines = f.readlines()
        
    USERNAME = lines[0][:-1]    #SERVIDOR
    PASSWORD = lines[1][:-1]
  
    #USERNAME = lines[0][1:-1]    #MINHA MAQUINA
    #PASSWORD = lines[1][:-1]
  
    exist_path = os.path.exists(os.path.join(base_path, 'data/CMEMS_forecast_daily_data'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/CMEMS_forecast_daily_data'))


    #Get the data in the format we want: data always at 12:30
    day_txt = (date.today() - timedelta(days=2)).strftime('%Y-%m-%d')
    date_motu_txt = day_txt + ' 12:30:00'

    OUTPUT_FILENAME = 'CMEMS_forecast_' + day_txt +'.nc'
    OUTPUT_DIRECTORY = 'projects/JUNO/data/CMEMS_forecast_daily_data'

    script_template = f'python -m motuclient \
        --motu https://nrt.cmems-du.eu/motu-web/Motu \
        --service-id IBI_ANALYSISFORECAST_PHY_005_001-TDS \
        --product-id cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m \
        --longitude-min -19 --longitude-max -5 \
        --latitude-min 35 --latitude-max 45 \
        --date-min "{date_motu_txt}" --date-max "{date_motu_txt}" \
        --variable thetao \
        --out-dir <OUTPUT_DIRECTORY> \
        --out-name <OUTPUT_FILENAME> \
        --user <USERNAME> --pwd <PASSWORD>'
    
    
    data_request_options_dict_automated = motu_option_parser(script_template, USERNAME, PASSWORD, OUTPUT_FILENAME, OUTPUT_DIRECTORY)
    #print(data_request_options_dict_automated)

    #Submit data request
    motuclient.motu_api.execute_request(MotuOptions(data_request_options_dict_automated))
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/CMEMS_forecast_daily_fronts_netcdf'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/CMEMS_forecast_daily_fronts_netcdf'))
  
    xarray_cmems_forecast = get_data('CMEMS_forecast_' + day_txt + '.nc', base_path=base_path)
    
    canny_front = canny_front_detection_1day(xarray_cmems_forecast, day_txt, base_path=base_path)
    
    boa_front = BOA_aplication(xarray_cmems_forecast, day_txt, base_path=base_path)
    
    cca_front = CCA_front(xarray_cmems_forecast, day_txt, base_path=base_path)
        
    sst = real_sst_image(xarray_cmems_forecast, day_txt, base_path=base_path)
    
    
    ################################################### CREATION OF THE NETCDF   #######################################################
    
    nc_file = os.getcwd()
    nc_file = os.path.join(nc_file, 'projects/JUNO/data/CMEMS_forecast_daily_fronts_netcdf/CMEMS' + day_txt + '.nc')

    ds = nc.Dataset(nc_file, 'w', format='NETCDF4')

    ds.title = 'MUR ' + day_txt + ' Fronts Arrays (Xarrays)'

    #create dimensions of the NetCDF file
    time = ds.createDimension('time')
    lat = ds.createDimension('lat', 361)
    lon = ds.createDimension('lon', 505)

    times = ds.createVariable('time', 'f4', ('time', ))
    lats = ds.createVariable('lat', 'f4', ('lat', ))
    lons = ds.createVariable('lon', 'f4', ('lon', ))

    sst_analyzed = ds.createVariable('sst', 'f4', ('time', 'lat', 'lon',))
    sst_analyzed.units = 'C'   #degrees Celsius
    sst_analyzed.description = 'Array with the Sea-Surface Temperature (SST) relative to the MUR data for that day'
    sst_analyzed[0, :, :] = sst

    canny = ds.createVariable('Canny', 'f4', ('time', 'lat', 'lon',))
    canny.units = 'Unknown'
    canny.description = 'Binary Array with identyfied fronts through Canny from OpenCV (1-> front), (0->not front)'
    canny[0, :, :] = canny_front.astype(float)
    
    boa = ds.createVariable('BOA', 'f4', ('time', 'lat', 'lon',))
    boa.units = 'Unknown'
    boa.description = 'Binary Array with identyfied fronts through the Belkin O Reilly Algorithm (temperature gradient). If the gradient is bigger than certain threshold is considered front (1) otherwise 0'
    boa[0, :, :] = boa_front
    
    cca = ds.createVariable('CCA', 'f4', ('time', 'lat', 'lon',))
    cca.units = 'Unknown'
    cca.description = 'Binary Array with identyfied fronts through the Cayula Cornillon Algorithm (1->front) (0->not front)'
    cca[0, :, :] = cca_front.astype(float)
    
    times.units = 'days since 1-1-1'

    lats[:] = np.linspace(35, 45, 361)
    lons[:] = np.linspace(-19, -5, 505)
   
   
    date_obj = datetime.datetime.strptime(day_txt, '%Y%m%d')
    date_time = date_obj.toordinal()
    times[:] = date_time

    ds.close()
    
    

if __name__ == "__main__":
    main()


