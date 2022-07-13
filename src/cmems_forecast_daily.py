
import motuclient
from datetime import date, timedelta
import numpy as np
import xarray as xr
import os
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import cmocean

matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import BOA
import CayulaCornillon


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
    Function to get our netCDF file that is stored in the data directory and convert it to a dataframe.
    The data parameter is the string name of the netCDF file we want to import
    """
    
    base_path = base_path
    data_folder = os.path.join(base_path, "data/CMEMS_forecast_daily_data")           
    
    nc_path = os.path.join(data_folder, data)
    netCDF = xr.load_dataset(nc_path)
    
    df = netCDF.to_dataframe()
    df = df.reset_index() 
        
    return df


#################################### CANNY ALGORITHM ##########################################################

def canny_front_detection_1day(df, day_txt, base_path, thresh_min=120, thresh_max=220, apertureSize=5, sigma=5):
    
    """
    This function receives a dataframe with MUR data for a individual day and plots the result
    of the aplication of the Canny Algorithm from OpenCV. 
    For visualization purposes, one can change the minimum and maximum threshold.
    One can also apply a gaussian filter with a certain sigma value to reduce noise of the image
    """

    # Convert the df to a numpy array with the SST values for the coordinate pair (longitude and latitude)
    sst = df.pivot_table(index='longitude', columns='latitude', values='thetao').T.values
    #Convert Temperature values to uint8 format with values in the range of 0-255
    sst_final = ((sst - np.nanmin(sst)) * (1/(np.nanmax(sst) - np.nanmin(sst)) * 255)).astype('uint8')
    sst_final = np.flipud(sst_final)   #flipud -> Reverse the order of elements along axis 0 (up/down).
    #in case we want to apply a gaussian filter with a certain sigma value (by default is 0)
    sst_final = gaussian_filter(sst_final, sigma=sigma)   

    #to define the extent of the plot
    lat = df['latitude'].to_numpy()   
    lon = df['longitude'].to_numpy()
    lat = np.unique(lat).round(3)
    lon = np.unique(lon).round(3)
    
    #apply the canny algorithm and plot the image with the edges
    canny = cv2.Canny(sst_final, thresh_min, thresh_max, apertureSize=apertureSize, L2gradient=False)

    #Apply a mask for the continental zone:
    mask = np.isnan(np.flipud(sst))    #Boolean array: True where array Temp had Null Values (correspond to the continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which values= 255 when mask=True
    #Dilation to ensure that the pixels that belong to the "shore/continental zone" are not considered fronts 
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    canny_mask =np.ma.masked_where(mask_dilated==255, canny)   #Mask an array where a condition is me
    
    #Visualization purposes: continenal area in gray, and pixels with value=0 in white   
    viridis = matplotlib.cm.get_cmap('viridis', 1)
    newcolor = viridis(np.linspace(0,1,10))
    white = np.array([1, 1, 1, 1])
    newcolor[0, :] = white
    newcmp = ListedColormap(newcolor)
    newcmp.set_bad(color='gray')
    
    plt.figure()
    plt.imshow(canny_mask, cmap=newcmp, extent=[lon[0], lon[-1], lat[0], lat[-1]])
    plt.title("Canny Algorithm (CMEMS Forecast) " + day_txt, fontsize=12)
    plt.savefig(os.path.join(base_path,'data/CMEMS_forecast_daily_images/Canny_' + day_txt +'.jpg'))


########################################################################################################################

################################### Belkin O'Reilly Algorithm ##########################################################


def BOA_aplication(df, day_txt, base_path):  
    
    """
    Function to, for a given dataframe with a longitude, latitude and SST columns, 
    identifies fronts through the application of BOA algorithm.
    We also need to define a threshold value to later get the frontal probabilities matrix
    (if the pixel value is greater than the threshold, then it is considered a front, otherwise don't). 
    """
    
    lat = np.array(df['latitude'].unique())
    lon = np.array(df['longitude'].unique())
    ingrid = np.array(df['thetao']).reshape(len(lat), len(lon))
    
    front = BOA.boa(lon=lon, lat=lat, ingrid=ingrid, nodata = np.nan, direction = False)
    front = np.flip(front, axis=0)
    front = np.array([[front[j][i] for j in range(len(front))] for i in range(len(front[0])-1,-1,-1)])
    
    #Visualization purposes: continenal area in gray, and pixels with value=0 in white   
    viridis = matplotlib.cm.get_cmap('viridis', 1)
    newcolor = viridis(np.linspace(0,1,10))
    white = np.array([1, 1, 1, 1])
    newcolor[0, :] = white
    newcmp = ListedColormap(newcolor)
    newcmp.set_bad(color='gray')
    
    plt.figure()
    plt.imshow(front, cmap=newcmp, extent=[lon[0], lon[-1], lat[0], lat[-1]])
    #limits of x and y axis for the MUR data
    plt.xlim([-18.95, -5])
    plt.ylim([35.05, 45])
    plt.title("BOA (CMEMS Forecast) " + day_txt, fontsize=12)
    plt.savefig(os.path.join(base_path,'data/CMEMS_forecast_daily_images/BOA_' + day_txt +'.jpg'))
    
    
################################## CAYULA-CORNILLON ALGORITHM #################################################################
    

def CCA_front(df, day_txt, base_path): 
    
    """
    This function receives a dataframe with MUR data for a individual day and plots the result
    of the aplication of the Cayula-Cornillon Algorithm
    """
    
    front = np.zeros((361, 505))        #initialize a matrix of zeros. This shape is for the CMEMS Forecats data
        
    xdata_final, ydata_final = CayulaCornillon.CCA_SIED(df)       
    
    cols_x = np.array([])
    for value in xdata_final:                     #convert values in array x to the respective index in a (1001, 1401) matrix
        aux_x = (19+value)/0.027723                  #these numbers are relative to the CMEMS Forecast data
        cols_x = np.append(cols_x, aux_x)
    
    rows_y = np.array([])
    for value in ydata_final:                     #convert values in array y to the respective index in a (1001, 1401) matrix
        aux_y = (45-value)/0.0277                  #these numbers are relative to the CMEMS Forecast data
        rows_y = np.append(rows_y, aux_y)

    cols_x = np.round(cols_x)
    rows_y = np.round(rows_y)
            
    for i in range(len(cols_x)):   #it could also be len(rows_y)
        front[int(rows_y[i]), int(cols_x[i])] = front[int(rows_y[i]), int(cols_x[i])] + 1
        
    front[front != 0] = 1
    
    #Create a masked_array in order to get the continental zone well defined
    
    #Convert some df to a numpy array with the SST values for each value of longitude and latitude
    sst = df.pivot_table(index='longitude', columns='latitude', values='thetao').T.values   
    mask = np.isnan(np.flipud(sst))       #Boolean array=True where array Temp had Null values (continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    front = np.ma.masked_where(mask_dilated==255, front)   
    
    
    lat = np.array(df['latitude'].unique())
    lon = np.array(df['longitude'].unique())
    lat = np.unique(lat).round(3)
    lon = np.unique(lon).round(3)
    
    #Visualization purposes: continenal area in gray, and pixels with value=0 in white   
    viridis = matplotlib.cm.get_cmap('viridis', 1)
    newcolor = viridis(np.linspace(0,1,10))
    white = np.array([1, 1, 1, 1])
    newcolor[0, :] = white
    newcmp = ListedColormap(newcolor)
    newcmp.set_bad(color='gray')
    
    plt.imshow(front, cmap=newcmp, extent = [lon[0], lon[-1], lat[0], lat[-1]])    #interpolation='nearest'
    #extent is to define the extention of the x and y axis
    plt.title("Cayula-Cornillon Algorithm (CMEMS Forecast) " + day_txt, fontsize=12)
    plt.savefig(os.path.join(base_path,'data/CMEMS_forecast_daily_images/CCA_' + day_txt +'.jpg'))
    plt.close()
    
##########################################################################################################################################
    
 ################################# GET THE REAL SST IMAGE FROM THE CMEMS FORECAST DATASET ##############################################
    
    
def real_sst_image(df, day_txt, base_path):
        
    """
    Function to store the real sst image
    """
    
    sst = df.pivot_table(index='longitude', columns='latitude', values='thetao').T.values
    sst = np.flipud(sst)
    
    lat = np.array(df['latitude'].unique())
    lon = np.array(df['longitude'].unique())
    lat = np.unique(lat).round(3)
    lon = np.unique(lon).round(3)
    
    plt.figure()
    plt.imshow(sst, cmocean.cm.thermal, extent = [lon[0], lon[-1], lat[0], lat[-1]])
    plt.title("Real SST Image (CMEMS Forecast) " + day_txt, fontsize=12)
    plt.savefig(os.path.join(base_path,'data/CMEMS_forecast_daily_images/RealSST_' + day_txt+'.jpg'))
    
    

def main():
    
    #My Username and Password are stored in a .txt file stored in a data folder which belong to the gitignore
    with open('JUNO/data/copernicus_login.txt') as f:   #quando fizer clone para o servidor esta documento .txt vai ser ignorado
        lines = f.readlines()
        
    USERNAME = lines[0][1:-1]
    PASSWORD = lines[1][:-1]
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/CMEMS_forecast_daily_data'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/CMEMS_forecast_daily_data'))


    #Get the data in the format we want: data always at 12:30
    day_txt = (date.today() - timedelta(days=2)).strftime('%Y-%m-%d')
    date_motu_txt = day_txt + ' 12:30:00'

    OUTPUT_FILENAME = 'CMEMS_forecast_' + day_txt +'.nc'
    OUTPUT_DIRECTORY = 'JUNO/data/CMEMS_forecast_daily_data'

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

    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'JUNO')
    
    #Quando criar o cronjob para correr este script diariamente, este for desaparece e day passa a ser 1 (yesterday)
    day_txt = (date.today() - timedelta(days=2)).strftime('%Y-%m-%d')
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/CMEMS_forecast_daily_images'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/CMEMS_forecast_daily_images'))
  
    df_cmems_forecast = get_data('CMEMS_forecast_' + day_txt + '.nc', base_path=base_path)
    
    canny_front_detection_1day(df_cmems_forecast, day_txt, base_path=base_path)
    
    BOA_aplication(df_cmems_forecast, day_txt, base_path=base_path)
    
    CCA_front(df_cmems_forecast, day_txt, base_path=base_path)
        
    real_sst_image(df_cmems_forecast, day_txt, base_path=base_path)
    

if __name__ == "__main__":
    main()


