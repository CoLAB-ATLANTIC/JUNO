
# This script allows you to import MUR data stored in the MUR_seasonal_data directory and 
# then apply the 3 algorithms (CCA being all commented out because it takes too long to apply CCA to large amounts of data). 
# The results of applying the algorithms (Canny and BOA) are stored in the MUR_seasonal_images directory

#Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import cv2
import matplotlib
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
import BOA     
import CayulaCornillon_df
import time

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
    data_folder = os.path.join(base_path, "data/MUR_seasonal_data")  
    
    nc_path = os.path.join(data_folder, data)
    netCDF = xr.load_dataset(nc_path)
    
    df = netCDF.to_dataframe()
    df = df.reset_index()
    
    df = df.drop(['depth'], axis=1, errors='ignore') #drop the column 'depth' if exists: only exists in reanalysis
    
    df.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'time': 'time', 'analysed_sst':'thetao'}, inplace=True)
    df['thetao'] = df['thetao']-273.15   
    
    return df


def get_period(df):
    
    """
    Function that receives a dataframe that has SST data for different days 
    and returns a dictionaire of dataframes (one for each different day) (dict_df) and 
    an array with the different dates its possible to find in our dataframe (specificday)
    """
    
    specificday = [pd.Timestamp(dd).strftime("%Y-%m-%d %H:%M:%S") for dd in df['time'].unique()]
    specificday = np.array(specificday, dtype=np.object)
  
    #create a dictionary to store the data frames for each day
    dict_df = {elem : pd.DataFrame for elem in specificday}

    for key in dict_df.keys():
        dict_df[key] = df[:][df['time'] == key]
        
    return dict_df, specificday



########################################################################################################################
########################################  CANNY FRONTAL PROBABILITIES  #################################################

def canny_front_calc(dict_df, Tmin, Tmax, sigma=5, apertureSize=5):  
    
    """
    Function that receives a dataframe with SST data relative to a certain day and returns the front matrix 
    obtained due to the aplication of the Canny algorithm.
    For each image a Gaussian filter (with a certain sigma value) might be applied (depending on the data)
    Tmin and Tmax are the limits of the threshold and apertureSize is the size of the Sobel operator (default=3X3)
    """
    
    #Convert the df to a numpy array with the SST values for the coordinate pair (longitude and latitude)
    Temp = dict_df.pivot_table(index='longitude', columns='latitude', values='thetao').T.values
    
    #Convert the temperature values to the uint8 format with values between 0-255
    Temp_day = ((Temp - np.nanmin(Temp)) * (1/(np.nanmax(Temp) - np.nanmin(Temp)) * 255)).astype('uint8')

    Temp_day = np.flipud(Temp_day)   #flipud -> Reverse the order of elements along axis 0 (up/down).
    
    #if its MUR data we have to apply gaussian filter with certain sigma value (~5)
    Temp_day = gaussian_filter(Temp_day, sigma=sigma)
    
    #apply the canny algorithm 
    canny = cv2.Canny(Temp_day, Tmin, Tmax, L2gradient=False, apertureSize=apertureSize)
    
    return canny  #return the matrix (if a pixel was considered a front than its value is 255; otherwise is 0)


def frontal_prob_canny(period, dict_df, Tmin, Tmax, sigma=5, apertureSize=5):
    """
    This function receives several front matrices and for that period calculates the frontal_probability. 
    Then it creates a masked_array so that the continental zone is well defined.
    This masked_array is applied to the frontal probabilities matrix, which is returned
    """
    
    fp = np.zeros((1001,1401))   #if its MUR data, fp shape must be (1001, 1401)
        
    for day in period:             #for each day in period we sum fp with the matrix resulted from canny
        fp = fp + canny_front_calc(dict_df[day], Tmin=Tmin, Tmax=Tmax, sigma=sigma, apertureSize=apertureSize)
    
    fp = fp/(len(period)*255)*100    # to get the fp in percentage 
    
    return fp 


def canny_frontal_prob_visualization(base_path, period_txt, period, dict_df, fp_canny, vmin=None, vmax=None):
    
    """
    Function to visualize the map of frontal probability.
    vmin and vmax define the data range that the colormap covers -> it helps for visualization purposes.
    """
    
    #first we apply the frontal_prob function to a certain period of data
    fp = fp_canny
    
    #Create a mask for our continental zone
    sst = dict_df[period[0]].pivot_table(index='longitude', columns='latitude', values='thetao').T.values  
    mask = np.isnan(np.flipud(sst))    #Boolean array is True where original array (Temp) had Null values
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    
    fp = np.ma.masked_where(mask_dilated==255, fp)
    
    #for the definition of the extent in the imshow() -> so we see the values of long and latitude in our plot
    lat = np.array(dict_df[period[0]]['latitude'].unique())
    lon = np.array(dict_df[period[0]]['longitude'].unique())
    
    #Visualization purposes: continenal area in gray, and pixels with value=0 in white   
    viridis = matplotlib.cm.get_cmap('viridis', 100)
    newcolor = viridis(np.linspace(0,1,100))
    white = np.array([1, 1, 1, 1])
    newcolor[0, :] = white
    newcmp = ListedColormap(newcolor)
    newcmp.set_bad(color='gray')

    plt.figure()
    plt.imshow(fp, cmap=newcmp, vmin=vmin, vmax=vmax, extent=[lon[0], lon[-1], lat[0], lat[-1]]) 
    plt.colorbar(orientation='horizontal', fraction=0.025, pad=0.08, aspect=50)
    plt.title("CANNY Frontal Probabilities (MUR) " + period_txt, fontsize=20)
    plt.savefig(os.path.join(base_path, 'data/MUR_seasonal_images/CANNY_frontal_prob_' + period_txt +'.jpg'))



################################################################################################################################
##################################### BOA FRONTAL PROBABILITIES   ###############################################################

def BOA_aplication(df, threshold=0.05):  
    
    """
    Function to, for a given dataframe with a longitude, latitude and SST columns, 
    identifies fronts through the application of BOA algorithm.
    We also need to define a threshold value to later get the frontal probabilities matrix
    (if the pixel value is greater than the threshold, then it is considered a front, otherwise its not). 
    """
    
    lat = np.array(df['latitude'].unique())
    lon = np.array(df['longitude'].unique())
    ingrid = np.array(df['thetao']).reshape(len(lat), len(lon))
    
    front = BOA.boa(lon=lon, lat=lat, ingrid=ingrid, nodata = np.nan, direction = False)
    front = np.flip(front, axis=0)
    front = np.array([[front[j][i] for j in range(len(front))] for i in range(len(front[0])-1,-1,-1)])
    
    front = np.where(front>=threshold, 1, front)    
    front = np.where(front<threshold, 0, front)
    
    return front


def frontal_prob_boa(period, df, threshold=0.05):
    
    """
    Function applies BOA to several images and returns the matrix of frontal probabilities for certain period.
    The matrices resulting from the application of BOA are summed and divided by the number of periods
    to obtain a front probabilities matrix.
    """
    
    fp = np.zeros((1001,1401))    
        
    for day in period:
        fp = fp + BOA_aplication(df[day], threshold=threshold)
    
    fp = fp/(len(period))*100     #for the calculation of the FP we divide by the number of periods (days) 
    
    return fp


def boa_frontal_prob_visualization(base_path, period, df, period_txt, fp_boa, vmin=None, vmax=None):
    
    """
    Function to visualize the frontal probabilities map.
    """
    
    fp = fp_boa
    
    lat = np.array(df[period[0]]['latitude'].unique())
    lon = np.array(df[period[0]]['longitude'].unique())
    
    #Visualization purposes: continenal area in gray, and pixels with value=0 in white   
    viridis = matplotlib.cm.get_cmap('viridis', 100)
    newcolor = viridis(np.linspace(0,1,100))
    white = np.array([1, 1, 1, 1])
    newcolor[0, :] = white
    newcmp = ListedColormap(newcolor)
    newcmp.set_bad(color='gray')
    
    plt.figure()
    plt.imshow(fp, cmap=newcmp, vmin=vmin, vmax=vmax, extent=[lon[0], lon[-1], lat[0], lat[-1]], interpolation='gaussian')
    #extent is to define the extension of x and y axis in image
    plt.xlim([-18.95, -5])
    plt.ylim([35.05, 45])
    plt.colorbar(orientation='horizontal', fraction=0.025, pad=0.08, aspect=50)
    plt.title("BOA Frontal Probabilities (MUR) " + period_txt, fontsize=20)
    plt.savefig(os.path.join(base_path, 'data/MUR_seasonal_images/BOA_frontal_prob_' + period_txt +'.jpg'))

    


####################################################################################################################################
############################################ CCA Frontal Probabilities  ############################################################

#def front_calc(df): 
    
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
        
    xdata_final, ydata_final = CayulaCornillon_df.CCA_SIED(df)       
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


#def frontal_prob_cca(period, dict_df):
    
    """
    Function that allows the visualization of the Frontal Probabilities for the Cayula-Cornillon Algorithm (CCA).
    It receives 2 parameters: period (its an array with the several string dates for the period in question) and
    dict_df (which is a dictionaire of dataframes) with data related to those days.
    This function also applies a mask to the frontal probabilities array in order for the continental zone to be 
    well defined. The function plots the frontal probabilities
    """
    front_prob = np.zeros((1001, 1401))    #for the resolution of the MUR data
    for day in period:
        front = front_calc(dict_df[day])
        
        front_prob = front_prob + front
        
    front_prob = front_prob/(len(period))*100    

    return front_prob
    
    
#def CCA_frontal_prob_visualization(base_path, period, dict_df, period_txt, fp_cca, vmax=None):   
    
    """
    The purpose of this function is to load the memory from different front matrixes for different days,
    calculate the front probability matrix for the period in question
    and make a visual demonstration of this matrix.
    """
    
    front_prob = fp_cca
    
    #Create a masked_array in order to get the continental zone well defined
    #Convert some df to a numpy array with the SST values for each value of longitude and latitude
    sst = dict_df[period[0]].pivot_table(index='longitude', columns='latitude', values='thetao').T.values    
    mask = np.isnan(np.flipud(sst))       #Boolean array=True where array Temp had Null values (continental zone)
    mask255 =np.where(mask,(np.ones(mask.shape))*255,0).astype("uint8")   #array which pixels = 255 when mask=True 
    #Make a dilation to ensure the pixels that belong to the shore are not consideredd fronts
    kernel = np.ones((3,3), np.uint8)
    mask_dilated = cv2.dilate(mask255, kernel)
    front_prob = np.ma.masked_where(mask_dilated==255, front_prob)   
       
    lat = np.array(dict_df[period[0]]['latitude'].unique())
    lon = np.array(dict_df[period[0]]['longitude'].unique())
    lat = np.unique(lat).round(3)
    lon = np.unique(lon).round(3)
    
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
    plt.colorbar(orientation='horizontal', fraction=0.025, pad=0.08, aspect=50)
    plt.savefig(os.path.join(base_path, 'data/MUR_seasonal_images/CCA_frontal_prob_' + period_txt +'.jpg'))
    

############################################################################################################################################################    
############################################################################################################################################################   
    
def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'JUNO')
    
    period_txt = input("Type name of the period for which we are applying the algorithms: ")
    
    fp_canny = np.zeros((1001,1401))   #if its MUR data, fp shape must be (1001, 1401)
    fp_boa = np.zeros((1001, 1401))
    #fp_cca = np.zeros((1001, 1401))
    count=0
    
    for filename in os.listdir((os.path.join(base_path, 'data/MUR_seasonal_data'))):
        
        df_mur = get_data(base_path=base_path, data = filename)    #neste caso data vai ser o nome do netcdf file que queremos importar (guardado no directorio MUR_seasonal_data)
        dict_df_mur, specificday_mur = get_period(df_mur)
        #dict_df_mur -> dictionaire of dataframes for each day of the period in question
        #specificday_mur -> array with all the days of the period in question
        
        start_time_canny = time.time()
        fp_canny = fp_canny + frontal_prob_canny(period=specificday_mur, dict_df=dict_df_mur, Tmin=200, Tmax=300, sigma=5, apertureSize=5)
        
        print(f'It took {filename}, {time.time()-start_time_canny} seconds to apply the Canny Algorithm')
        
        start_time_boa = time.time()
        fp_boa = fp_boa + frontal_prob_boa(period=specificday_mur, df=dict_df_mur, threshold=0.05)
        
        print(f'It took {filename}, {time.time()-start_time_boa} seconds to apply the BOA')
        
        #start_time_cca = time.time()
        #fp_cca = fp_cca + frontal_prob_cca(period=specificday_mur, dict_df=dict_df_mur)
        
        #print(f'It took {filename}, {time.time()-start_time_cca} seconds to apply the CCA')
        
        count += 1
        
    fp_canny = fp_canny/count
    fp_boa = fp_boa/count
    #fp_cca = fp_cca/count
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_seasonal_images'))
    if not exist_path:
        os.makedirs(os.path.join(base_path, 'data/MUR_seasonal_images'))
    
    canny_frontal_prob_visualization(base_path=base_path, period_txt=period_txt, period=specificday_mur, dict_df=dict_df_mur, fp_canny=fp_canny, vmin=None, vmax=None)
    
    boa_frontal_prob_visualization(base_path=base_path, period=specificday_mur, df=dict_df_mur, period_txt=period_txt, fp_boa=fp_boa, vmin=None, vmax=None)
    
    #CCA_frontal_prob_visualization(base_path=base_path, period=specificday_mur, dict_df=dict_df_mur, period_txt=period_txt, fp_cca=fp_cca, vmax=None)
    
    

if __name__ == "__main__":
    main()