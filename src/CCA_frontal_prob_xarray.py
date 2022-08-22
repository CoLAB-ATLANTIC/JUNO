#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import cv2
import matplotlib
from matplotlib.colors import ListedColormap
#import CayulaCornillon
import CayulaCornillon_xarray
import time
import gc
import sys

matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import warnings
warnings.filterwarnings("ignore")



######################################### IMPORT DATA #######################################################


def get_data(base_path, data):
    
    """
    Function to get our netCDF file that is stored in the data directory inside the MUR_seasonal_data folder and convert it to a dataframe.
    The data parameter is the string name of the netCDF file we want to import
    """
    
    base_path = base_path
    data_folder = os.path.join(base_path, "data/MUR_single_days")  
    
    nc_path = os.path.join(data_folder, data)
    data_xarray = xr.load_dataset(nc_path)
    
    return data_xarray



####################################################################################################################################
############################################ CCA Frontal Probabilities  ############################################################

def front_calc(data_xarray): 
    
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
    
    return front    

    
    
def CCA_frontal_prob_visualization(base_path, data_xarray, period_txt, fp_cca, vmax=None):   
    
    """
    The purpose of this function is to load the memory from different front matrixes for different days,
    calculate the front probability matrix for the period in question
    and make a visual demonstration of this matrix.
    """
    
    front_prob = fp_cca
    
    #Create a masked_array in order to get the continental zone well defined
    #Convert xarray to a numpy array with the SST values for each value of longitude and latitude 
    sst = data_xarray['analysed_sst'][0].values - 273.15
    mask = np.isnan(np.flipud(sst))       #Boolean array=True where array Temp had Null values (continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    front_prob = np.ma.masked_where(mask_dilated==255, front_prob)   
    
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
    plt.imshow(front_prob, cmap=newcmp, extent = [lon[0], lon[-1], lat[0], lat[-1]], interpolation='bilinear', vmax=vmax)    
    #extent is to define the extention of the x and y axis
    plt.title("Cayula-Cornillon Algorithm Frontal Probability (MUR) " + period_txt, fontsize=20)
    #plt.colorbar(orientation='horizontal', fraction=0.025, pad=0.08, aspect=50)
    plt.savefig(os.path.join(base_path, 'data/MUR_seasonal_images/CCA_frontal_prob_' + period_txt +'.jpg'))
    

############################################################################################################################################################    
############################################################################################################################################################   
    
def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'JUNO')
    
    #period_txt = input("Type name of the period for which we are applying the algorithms: ")
    
    #fp_cca = np.zeros((1001, 1401))
    #count=0
    
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_daily_fronts_npy'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/MUR_daily_fronts_npy'))
    
    
    #for filename in sorted(os.listdir((os.path.join(base_path, 'data/MUR_single_days')))):
    
    filename = os.path.join(base_path, 'data/MUR_single_days', sys.argv[1])
        
    start_time_cca = time.time()
        
    data_xarray = get_data(base_path=base_path, data = filename)    #neste caso data vai ser o nome do netcdf file que queremos importar (guardado no directorio MUR_seasonal_data)
        
    front = front_calc(data_xarray)
    front = front.astype('int8')
    
    #fp_cca = fp_cca + front
    
    
    np.save('JUNO/data/MUR_daily_fronts_npy/' + filename.replace('nc', 'npy').split('/')[-1], front)
    
    del(front)
    del(data_xarray)
    gc.collect()
    
    #count += 1
    
    print(f'It took {filename}, {time.time()-start_time_cca} seconds to get the netCDF from file, convert it to an xarray, apply the CCA and save the fronts array')
    
    #count += 1
        
    #so serve para a função CCA_frontal_prob_visualization para fazer um map da zona continental
    #data_xarray = get_data(base_path=base_path, data='sst_20190601.nc')
        

    #fp_cca = fp_cca/count
    
    #exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_seasonal_images'))
    #if not exist_path:
     #   os.makedirs(os.path.join(base_path, 'data/MUR_seasonal_images'))

    
    #CCA_frontal_prob_visualization(base_path=base_path, data_xarray = data_xarray, period_txt=period_txt, fp_cca=fp_cca, vmax=None)
    
    

if __name__ == "__main__":
    main()