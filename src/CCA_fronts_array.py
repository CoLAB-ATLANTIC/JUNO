

################################################### CCA_fronts_array.py  ####################################################
###                                                                                                                       ###
###    imports data from the MUR_single_days folder (that have netCDF files with all the days from 2012 to 2021)          ###
###    Then applies the Cayula-Cornillon Algorithm, which returns a np array of fronts for each of the single days        ###
###    Then those numpy arrays are stored in a MUR_daily_fronts_npy folder                                                ###
###                                                                                                                       ###
#############################################################################################################################


#Import Libraries
import numpy as np
import xarray as xr
import os
import matplotlib
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

    

############################################################################################################################################################    
############################################################################################################################################################   
    
def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'JUNO')
    
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_daily_fronts_npy'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/MUR_daily_fronts_npy'))
    
    #lemos todas as datas através do shell script front_arrays.sh 
    filename = os.path.join(base_path, 'data/MUR_single_days', sys.argv[1])
        
    start_time_cca = time.time()
        
    data_xarray = get_data(base_path=base_path, data = filename)   
        
    front = front_calc(data_xarray)
    front = front.astype('int8')
    
    
    np.save('JUNO/data/MUR_daily_fronts_npy/' + filename.replace('nc', 'npy').split('/')[-1], front)
    
    del(front)
    del(data_xarray)
    gc.collect()

    
    print(f'It took {filename}, {time.time()-start_time_cca} seconds to get the netCDF from file, convert it to an xarray, apply the CCA and save the fronts array')
    


if __name__ == "__main__":
    main()