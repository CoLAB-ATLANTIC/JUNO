

################################################### CMEMS_daily_image.py  ##########################################################
###                                                                                                                              ###
###    Through a cron job gets the daily data from the CMEMS Forecast (with 2 days delay but that can be changed)                ###
###    that is located in the data folder: CMEMS_forecast_daily_data.                                                            ###
###    Applies the 3 algorithms, and saves the results (png image for each algorithm) in the CMEMS_forecast_daily_images         ###
###                                                                                                                              ###
####################################################################################################################################



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
import CayulaCornillon_xarray    #CayulaCornillon after making some profiling changes to improve efficiency



######################################### IMPORT DATA ################################################################

def get_data(data, base_path):
    
    """
    Function to get our netCDF file that is stored in the data directory inside the MUR_seasonal_data folder and convert it to a dataframe.
    The data parameter is the string name of the netCDF file we want to import
    """
    
    base_path = base_path
    data_folder = os.path.join(base_path, "data/CMEMS_daily_data")  
    
    nc_path = os.path.join(data_folder, data)
    data_xarray = xr.load_dataset(nc_path)
    
    #para estarem todos com a mesma nomenclatura usada o CCA_SIED do script CayulaCornillon_xarray.py
    data_xarray = data_xarray.rename({'latitude':'lat', 'longitude':'lon', 'thetao':'analysed_sst'})
    
    return data_xarray


#################################### CANNY ALGORITHM ##########################################################

def canny_front_detection_1day(data_xarray, day_txt, base_path, thresh_min=120, thresh_max=220, apertureSize=5, sigma=5):
    
    """
    This function receives a dataframe with MUR data for a individual day and plots the result
    of the aplication of the Canny Algorithm from OpenCV. 
    For visualization purposes, one can change the minimum and maximum threshold.
    One can also apply a gaussian filter with a certain sigma value to reduce noise of the image
    """

    sst = np.array(data_xarray['analysed_sst'])
    sst = np.squeeze(sst)
    #Convert Temperature values to uint8 format with values in the range of 0-255
    sst_final = ((sst - np.nanmin(sst)) * (1/(np.nanmax(sst) - np.nanmin(sst)) * 255)).astype('uint8')
    sst_final = np.flipud(sst_final)   #flipud -> Reverse the order of elements along axis 0 (up/down).
    #in case we want to apply a gaussian filter with a certain sigma value (by default is 0)
    sst_final = gaussian_filter(sst_final, sigma=sigma)   

    #to define the extent of the plot
    lon = np.array(data_xarray['lon']).astype('float64')
    lat = np.array(data_xarray['lat']).astype('float64')
    
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


def BOA_aplication(data_xarray, day_txt, base_path):  
    
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
    

def CCA_front(data_xarray, day_txt, base_path): 
    
    """
    This function receives a dataframe with MUR data for a individual day and plots the result
    of the aplication of the Cayula-Cornillon Algorithm
    """
    
    front = np.zeros((361, 505))        #initialize a matrix of zeros. This shape is for the CMEMS Forecats data
        
    xdata_final, ydata_final = CayulaCornillon_xarray.CCA_SIED(data_xarray)       
    
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
      
    sst = np.array(data_xarray['analysed_sst'])
    sst = np.squeeze(sst)
    mask = np.isnan(np.flipud(sst))       #Boolean array=True where array Temp had Null values (continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    front = np.ma.masked_where(mask_dilated==255, front)   
    
    
    #to define the extent of the plot
    lon = np.array(data_xarray['lon']).astype('float64')
    lat = np.array(data_xarray['lat']).astype('float64')
    
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
    
    
def real_sst_image(data_xarray, day_txt, base_path):
        
    """
    Function to store the real sst image
    """

    sst = np.array(data_xarray['analysed_sst'])
    sst = np.squeeze(sst)
    sst = np.flipud(sst)
    
    #to define the extent of the plot
    lon = np.array(data_xarray['lon']).astype('float64')
    lat = np.array(data_xarray['lat']).astype('float64')
    
    plt.figure()
    plt.imshow(sst, cmocean.cm.thermal, extent = [lon[0], lon[-1], lat[0], lat[-1]])
    plt.title("Real SST Image (CMEMS Forecast) " + day_txt, fontsize=12)
    plt.savefig(os.path.join(base_path,'data/CMEMS_forecast_daily_images/RealSST_' + day_txt+'.jpg'))
    
    
###############################################################################################################################################################
    

def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'projects/JUNO')       #servidor
    #base_path = os.path.join(base_path, 'JUNO')                #minha maquina
    

    #Get the data in the format we want: data always at 12:30
    day_txt = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/CMEMS_forecast_daily_images'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/CMEMS_forecast_daily_images'))
  
    xarray_cmems_forecast = get_data('CMEMS_' + day_txt + '.nc', base_path=base_path)
    
    canny_front_detection_1day(xarray_cmems_forecast, day_txt, base_path=base_path)
    
    BOA_aplication(xarray_cmems_forecast, day_txt, base_path=base_path)
    
    CCA_front(xarray_cmems_forecast, day_txt, base_path=base_path)
        
    real_sst_image(xarray_cmems_forecast, day_txt, base_path=base_path)
    

if __name__ == "__main__":
    main()


