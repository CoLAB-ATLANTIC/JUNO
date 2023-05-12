
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
from scipy.ndimage import gaussian_filter
from datetime import date, timedelta, datetime
import wget
from math import floor
from pydap.client import open_url
import netCDF4 as nc

matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import BOA
import CayulaCornillon_xarray


################################################ DOWNLOAD MUR DATA ##################################################

import dask
import requests
#Allows us to visualize the dask progress for parallel operations
from dask.diagnostics import ProgressBar
ProgressBar().register()

import netrc
from subprocess import Popen
from platform import system
from getpass import getpass

import urllib
from urllib import request, parse
from http.cookiejar import CookieJar
import json


def setup_earthdata_login_auth(endpoint):
    """
    Set up the request library so that it authenticates against the given Earthdata Login
    endpoint and is able to track cookies between requests.  This looks in the .netrc file
    first and if no credentials are found, it prompts for them.
    Valid endpoints include:
        urs.earthdata.nasa.gov - Earthdata Login production
    """
    try:
        username = os.environ.get("MUR_USERNAME")
        password = os.environ.get("MUR_PASSWORD")
        
        #username, _, password = netrc.netrc().authenticators(endpoint)
    except (FileNotFoundError, TypeError):
        # FileNotFound = There's no .netrc file
        # TypeError = The endpoint isn't in the netrc file, causing the above to try unpacking None
        print('Please provide your Earthdata Login credentials to allow data access')
        print('Your credentials will only be passed to %s and will not be exposed in Jupyter' % (endpoint))
        username = input('Username:')
        password = getpass.getpass()

    
    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, endpoint, username, password)
    auth = request.HTTPBasicAuthHandler(manager)

    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)
    
    
    
def request_data(date_str):
    
    """
    Function to request the MUR data. date_str is a string in the format: %Y-%m-%d
    The data will be downloaded to the MUR_daily_data folder which is inside the data folder
    We return the basename to know which netcdf file we just donwloaded to the MUR_daily_data folder
    """
    date_string = date_str + 'T09:00:00Z'
    
    url = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C1996881146-POCLOUD&temporal=' + date_string + ',' + date_string + '&pageSize=365'
    
    #CMR Link to use
    #https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C1625128926-GHRC_CLOUD&temporal=2019-01-01T10:00:00Z,2019-12-31T23:59:59Z
    r = requests.get(url)
    response_body = r.json()
    
    
    od_files = []
    for itm in response_body['items']:
        for urls in itm['umm']['RelatedUrls']:
            if 'OPeNDAP' in urls['Description']:
                od_files.append(urls['URL'])

    print(od_files)
    
    
    base_path = os.getcwd()
   # base_path = os.path.join(base_path, '../data')      #local machine
    
    base_path = os.path.join(base_path, 'projects/JUNO/data')    #SERVIDOR
    
    


    exist_path = os.path.exists(os.path.join(base_path, 'MUR_daily_data'))   #check if folder MUR_daily_data exists in data folder
    if not exist_path:                                                            #if it don't exist:
        os.makedirs(os.path.join(base_path, 'MUR_daily_data'))  


    for f in od_files:
        print (" opening " + f)
        data_url = f'{f}.dap.nc4'

        # The notation below is [start index, step, end index]
        # lat[ /lat= 0..17998] start index. = -90
        # lon[ /lon= 0..35999] start index. = -180
        # time[ /time= 0..0] 

        required_variables = {'analysed_sst[0:1:0][12499:1:13499][16099:1:17499]',
                              'lat[12499:1:13499]',     # [35, 45]
                              'lon[16099:1:17499]',     # [-19, -5]
                              'time[0:1:0]'}

        basename = os.path.basename(data_url)
        request_params = {'dap4.ce': ';'.join(required_variables)}
        #identity encoding to work around an issue with server side response compression (??)
        response = requests.get(data_url, params=request_params,  headers={'Accept-Encoding': 'identity'})

        #basename = os.path.join('../data/MUR_daily_data', basename)    #local machine
        
        basename = os.path.join('projects/JUNO/data/MUR_daily_data', basename)    #SERVIDOR
        
    

        if response.ok:
            with open(basename, 'wb') as file_handler:
                file_handler.write(response.content)
        else:
            print(f'Request failed: {response.text}')
            
            
        # Replace "-" with ""
        filename_new = 'sst_' + date_str.replace("-", "") + '.nc'
        #basepath_new = os.path.join('../data/MUR_daily_data', filename_new)    #Local Machine
        
        basepath_new = os.path.join('projects/JUNO/data/MUR_daily_data', filename_new)   #SERVIDOR
        
        os.rename(basename, basepath_new)
            
        return basepath_new





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
    xarray_mur['analysed_sst'] = xarray_mur['analysed_sst'] - 273.15
    
    return xarray_mur

#################################### CANNY ALGORITHM ##########################################################

def canny_front_detection_1day(data_xarray, thresh_min=120, thresh_max=220, apertureSize=5, sigma=5):
    
    """
    This function receives a dataframe with MUR data for a individual day and returns the array 
    that result from the aplication of the Canny Algorithm from OpenCV. 
    For visualization purposes, one can change the minimum and maximum threshold.
    One can also apply a gaussian filter with a certain sigma value to reduce noise of the image.
    """
    
    #Get the sst array in the right shape
    sst = data_xarray['analysed_sst'][0,:,:].values
    #sst = np.array(data_xarray['analysed_sst'])
    #sst = np.squeeze(sst)
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
    #sst = data_xarray['analysed_sst'][0,:,:].values
    #mask = np.isnan(np.flipud(sst))    #Boolean array: True where array Temp had Null Values (correspond to the continental zone)
    #mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which values= 255 when mask=True
    #Dilation to ensure that the pixels that belong to the "shore/continental zone" are not considered fronts 
    #kernel = np.ones((3,3), np.uint8)
    #mask_dilated = cv2.dilate(mask255, kernel)
    #canny_front = np.ma.masked_array(canny, mask_dilated)   #Mask an array where a condition is True
    
    canny_front = np.flipud(canny)    
    
    
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

    
    
    #Create a masked_array in order to get the continental zone well defined
    #sst = np.array(data_xarray['analysed_sst'])
    #sst = np.squeeze(sst)    #--> Em vez disto posso simplesmente usar o ingrid criado em cima
    #sst = data_xarray['analysed_sst'][0,:,:].values

    #mask = np.isnan(np.flipud(sst))       #Boolean array=True where array Temp had Null values (continental zone)
    #mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    #kernel = np.ones((3,3), np.uint8)
    #mask_dilated = cv2.dilate(mask255, kernel)
    #boa_front = np.ma.masked_array(boa_front, mask_dilated)  
    
    boa_front = np.flipud(boa_front) 
    
    #convert 0s to Nans
    #boa_front[boa_front == 0] = 'nan'

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


    #Create a masked_array in order to get the continental zone well defined
    
    #Convert some df to a numpy array with the SST values for each value of longitude and latitude
    #sst = np.array(data_xarray['analysed_sst'])
    #sst = np.squeeze(sst)
    #sst = data_xarray['analysed_sst'][0,:,:].values
    
    #mask = np.isnan(np.flipud(sst))       #Boolean array=True where array Temp had Null values (continental zone)
    #mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    #kernel = np.ones((3,3), np.uint8)
    #mask_dilated = cv2.dilate(mask255, kernel)
    #cca_front = np.ma.masked_array(front, mask_dilated)  
    
    cca_front = np.flipud(front) 
    
    #convert 0s to Nans
    #cca_front[cca_front == 0] = 'nan'
    
    return cca_front
    
    ##########################################################################################################################################
    
    ################################# GET THE REAL SST IMAGE FROM THE CMEMS FORECAST DATASET ##############################################
    
    
def real_sst_image(data_xarray):
        
    """
    Function to store the real sst image
    """
    
    sst_image = data_xarray['analysed_sst'][0,:,:].values
    #sst = np.squeeze(sst)
    #sst_image = np.flipud(sst_image)
    
    return sst_image
 


def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'projects/JUNO')      #servidor
    
    #base_path = os.path.join(base_path, '../')      #local machine
    
    #download MUR data for the day before yesterday
    day_txt = (date.today() - timedelta(days=2)).strftime('%Y%m%d')
        
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_daily_data'))   #check if folder MUR_dailyu_data exists in data folder
    if not exist_path:                                                            #if it don't exist:
        os.makedirs(os.path.join(base_path, 'data/MUR_daily_data'))               #create the folder

    #check if the daily sst data file already exists in the MUR_daily_data folder. If it does delete it  
    exist_sst_file = os.path.join(base_path, 'data/MUR_daily_data/sst_' + day_txt + '.nc')
    if os.path.exists(exist_sst_file):
        os.remove(exist_sst_file)
        


        
        
    # urs = 'urs.earthdata.nasa.gov'    # Earthdata URL endpoint for authentication
    # prompts = ['Enter NASA Earthdata Login Username: ',
    #         'Enter NASA Earthdata Login Password: ']

    # # Determine the OS (Windows machines usually use an '_netrc' file)
    # netrc_name = "_netrc" if system()=="Windows" else ".netrc"
    

    # # Determine if netrc file exists, and if so, if it includes NASA Earthdata Login Credentials
    # try:
    #     netrcDir = os.path.expanduser(f"~/{netrc_name}")
    #     netrc(netrcDir).authenticators(urs)[0]

    # # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
    # except FileNotFoundError:
    #     homeDir = os.path.expanduser("~")
    #     Popen('touch {0}{2} | echo machine {1} >> {0}{2}'.format(homeDir + os.sep, urs, netrc_name), shell=True)
    #     Popen('echo login {} >> {}{}'.format(getpass(prompt=prompts[0]), homeDir + os.sep, netrc_name), shell=True)
    #     Popen('echo \'password {} \'>> {}{}'.format(getpass(prompt=prompts[1]), homeDir + os.sep, netrc_name), shell=True)
    #     # Set restrictive permissions
    #     Popen('chmod 0600 {0}{1}'.format(homeDir + os.sep, netrc_name), shell=True)

    #     # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
    # except TypeError:
    #     homeDir = os.path.expanduser("~")
        
    #     Popen('echo machine {1} >> {0}{2}'.format(homeDir + os.sep, urs, netrc_name), shell=True)
    #     Popen('echo login {} >> {}{}'.format(getpass(prompt=prompts[0]), homeDir + os.sep, netrc_name), shell=True)
    #     Popen('echo \'password {} \'>> {}{}'.format(getpass(prompt=prompts[1]), homeDir + os.sep, netrc_name), shell=True)
            
        
        
        
        
    edl="urs.earthdata.nasa.gov"

    setup_earthdata_login_auth(edl)
    

    # Convert the date string to a datetime object
    date_object = datetime.strptime(day_txt, "%Y%m%d")
    # Convert the datetime object back to a formatted string
    formatted_date_string = datetime.strftime(date_object, "%Y-%m-%d")
        
    nc_path = request_data(date_str=formatted_date_string) 
    
    xarray_mur = get_data(nc_path)
        
        
    #breakpoint()
    sst_image = real_sst_image(xarray_mur)
    
    canny_front = canny_front_detection_1day(xarray_mur)
    
    boa_front = BOA_aplication(xarray_mur, threshold=0.05)
    
    cca_front = CCA_front(xarray_mur)
        

    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_daily_fronts_netcdf'))    #check if folder MUR_algorithm_daily_images exists in data folder
    if not exist_path:                                                                         #if doesn't exist
        os.makedirs(os.path.join(base_path, 'data/MUR_daily_fronts_netcdf'))                # create the folder
        
        
        
    ################################################### CREATION OF THE NETCDF   #######################################################

    nc_file = os.getcwd()
    nc_file = os.path.join(nc_file, 'projects/JUNO/data/MUR_daily_fronts_netcdf/MUR' + day_txt + '.nc')    #SERVIDOR  
    
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
