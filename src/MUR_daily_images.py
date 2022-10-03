################################################### MUR_daily_fronts_netcdf.py  #############################################
###                                                                                                                       ###
###    Through a cronjob the MUR daily data is downloaded and stored in the MUR_daily_data folder.                        ###
###    Then the 3 algorithms are applied to this data to get the image of the fronts (for the Canny, BOA and CCA)         ###
###    The 3 images plus the SST image are then storen in the MUR_algorithm_daily_images folder                           ###
###                                                                                                                       ###
#############################################################################################################################



#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import cmocean
import cv2
import matplotlib
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from datetime import date, timedelta

matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import BOA
import CayulaCornillon_xarray


######################################### IMPORT DATA #######################################################

def get_data(data, base_path):
    
    """
    Function to get our netCDF file that is stored in the data directory inside the MUR_seasonal_data folder and convert it to a dataframe.
    The data parameter is the string name of the netCDF file we want to import
    """
    
    base_path = base_path
    data_folder = os.path.join(base_path, "data/MUR_daily_data")  
    
    nc_path = os.path.join(data_folder, data)
    data_xarray = xr.load_dataset(nc_path)
    
    #In the case the temperature is in Kelvin (Convert to Celsius)
    data_xarray['analysed_sst'] = data_xarray['analysed_sst'] - 273.15
    
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
    plt.title("Canny Algorithm (MUR) " + day_txt, fontsize=25)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/Canny_' + day_txt +'.jpg'))


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
    plt.title("BOA (MUR) " + day_txt, fontsize=20)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/BOA_' + day_txt +'.jpg'))



################################## CAYULA-CORNILLON ALGORITHM #################################################################
    

def CCA_front(data_xarray, day_txt, base_path): 
    
    """
    This function receives a dataframe with MUR data for a individual day and plots the result
    of the aplication of the Cayula-Cornillon Algorithm
    """
    

    front = np.zeros((1001,1401))       #initialize a matrix of zeros. This shape is for the MUR data
        
    xdata_final, ydata_final = CayulaCornillon_xarray.CCA_SIED(data_xarray)       
    
    cols_x = np.array([])
    for value in xdata_final:                     #convert values in array x to the respective index in a (1001, 1401) matrix
        aux_x = (19+value)/0.01                  #these numbers are relative to the MUR data
        cols_x = np.append(cols_x, aux_x)
    
    rows_y = np.array([])
    for value in ydata_final:                     #convert values in array y to the respective index in a (1001, 1401) matrix
        aux_y = (45-value)/0.01                  #these numbers are relative to the MUR data
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
    plt.title("Cayula-Cornillon Algorithm (MUR) " + day_txt, fontsize=20)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/CCA_' + day_txt +'.jpg'))
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
    plt.title("Real SST Image (MUR) " + day_txt, fontsize=12)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/RealSST_' + day_txt+'.jpg'))
        


def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'projects/JUNO')      #servidor
   # base_path = os.path.join(base_path, 'JUNO')               #minha maquina
    
    day_txt = (date.today() - timedelta(days=2)).strftime('%Y%m%d')
        
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_algorithm_daily_images'))    #check if folder MUR_algorithm_daily_images exists in data folder
    if not exist_path:                                                                         #if doesn't exist
        os.makedirs(os.path.join(base_path, 'data/MUR_algorithm_daily_images'))                # create the folder
            
    
    xarray_mur = get_data('sst_' + day_txt + '.nc', base_path=base_path)
    
    canny_front_detection_1day(xarray_mur, day_txt, base_path=base_path)
    
    BOA_aplication(xarray_mur, day_txt, base_path=base_path)
    
    CCA_front(xarray_mur, day_txt, base_path=base_path)
        
    real_sst_image(xarray_mur, day_txt, base_path=base_path)
    

if __name__ == "__main__":
    main()

