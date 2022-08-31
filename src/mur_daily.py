#########################################     mur_daily.py    ############################################
    
##########################################################################################################


# Script which, through a daily cron job makes the donwload of daily MUR data and saves it in the MUR_daily_data folder.
# Then it applies the 3 algorithms to the data and stores the 3 images results + the real image in the MUR_algorithm_daily_images 


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
import wget
import sys
from math import floor
from tqdm import tqdm
from pydap.client import open_url

matplotlib.use('Agg')    #por causa do erro AttributeError: 'NoneType' object has no attribute 'set_cursor'

import BOA
import CayulaCornillon_df


################################################ DOWNLOAD MUR DATA ##################################################
def boundingindex(dmin, dint, boundary0, boundary1):
    """
    get boundaries values to download the data already cropped
    """
    inx0 = max(int(floor((boundary0 - dmin) / dint)), 0)
    inx1 = max(int(floor((boundary1 - dmin) / dint)), 0)
    if inx0 > inx1:
        atemp = inx0
        inx0 = inx1
        inx1 = atemp
    return [inx0, inx1]


def get_mur_params(lon_box, lat_box):
    """
    Check sample file to get parameters for sst
    """
    mur_ncin = open_url(
        'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1/2018/002/20180102090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
    lon = mur_ncin['lon']
    lat = mur_ncin['lat']
    lon_step = np.mean(np.diff(lon))
    lat_step = np.mean(np.diff(lat))
    [mur_i0, mur_i1] = boundingindex(lon[0][0].data, lon_step, lon_box[0], lon_box[1])
    [mur_j0, mur_j1] = boundingindex(lat[0][0].data, lat_step, lat_box[0], lat_box[1])

    return mur_i0, mur_i1, mur_j0, mur_j1


def download_from_url(fileget, filenameout, replace, printiti):
    """
    function that tries to download data from "fileget" if the data didn't previously exist,
    the user asked to replace the old data, or the file has 0 bytes
    """
    exists = os.path.exists(filenameout)
    if exists:
        file_size = os.path.getsize(filenameout)
    else:
        file_size = 1
    if (not exists) or (replace and exists) or (file_size == 0):
        if exists:
            os.remove(filenameout)
        try:
            return wget.download(fileget, out=filenameout, bar=None)
        except Exception as e:
            if printiti:
                print(e)
                print("Error downloading")
                print("Download error url: " + fileget)
            return e
    return filenameout

def download_sst(path, date, mur_j0, mur_j1, mur_i0, mur_i1, replace):
    
    """
    Function to download individual days of MUR data
    """
    
    opendap_dir = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1/' + str(
        date.year) + '/'
    filename = opendap_dir + "{0:0>3}".format(str(date.dayofyear)) + '/' + date.strftime(
        "%Y%m%d") + '090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc.nc4'
    filenameout = path + "sst_" + date.strftime("%Y%m%d") + '.nc'
    fileget = filename + '?analysed_sst[0:1:0][' + str(mur_j0) + ':1:' + str(mur_j1) + '][' + str(mur_i0) + ':1:' + str(
        mur_i1) + ']'
    download_from_url(fileget, filenameout, replace, 1)
    return


def download_sst_thread(data_range, sst_path, mur_j0, mur_j1, mur_i0, mur_i1, replace):
    """
    Function to download several MUR data days simultaneously
    """
    for date in tqdm(data_range, desc='SST', file=sys.stdout):
        download_sst(sst_path, date, mur_j0, mur_j1, mur_i0, mur_i1, replace)
    return


######################################### IMPORT DATA #######################################################

def get_data(data, base_path):
    
    """
    Function to get our netCDF file that is stored in the data directory and convert it to a dataframe.
    The data parameter is the string name of the netCDF file we want to import
    """
    
    base_path = base_path
    data_folder = os.path.join(base_path, "data/MUR_daily_data")           
    
    nc_path = os.path.join(data_folder, data)
    netCDF = xr.load_dataset(nc_path)
    
    df = netCDF.to_dataframe()
    df = df.reset_index()
    
    df = df.drop(['depth'], axis=1, errors='ignore') #drop the column 'depth' if exists: only exists in reanalysis
    
    #if we are importing MUR data, rename columns and convert temperature to Celsius
    if data.startswith('mur') or data.startswith('sst'):   #the downloaded individual day mur data name starts with sst
        df.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'time': 'time', 'analysed_sst':'thetao'}, inplace=True)
        df['thetao'] = df['thetao']-273.15   
        
    
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
    plt.title("Canny Algorithm (MUR) " + day_txt, fontsize=25)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/Canny_' + day_txt +'.jpg'))


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
    plt.title("BOA (MUR) " + day_txt, fontsize=20)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/BOA_' + day_txt +'.jpg'))



################################## CAYULA-CORNILLON ALGORITHM #################################################################
    

def CCA_front(df, day_txt, base_path): 
    
    """
    This function receives a dataframe with MUR data for a individual day and plots the result
    of the aplication of the Cayula-Cornillon Algorithm
    """
    

    front = np.zeros((1001,1401))       #initialize a matrix of zeros. This shape is for the MUR data
        
    xdata_final, ydata_final = CayulaCornillon_df.CCA_SIED(df)       
    
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
    plt.title("Cayula-Cornillon Algorithm (MUR) " + day_txt, fontsize=20)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/CCA_' + day_txt +'.jpg'))
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
    plt.title("Real SST Image (MUR) " + day_txt, fontsize=20)
    plt.savefig(os.path.join(base_path,'data/MUR_algorithm_daily_images/RealSST_' + day_txt+'.jpg'))
        


def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'projects/JUNO')      #servidor
   # base_path = os.path.join(base_path, 'JUNO')               #minha maquina
    
    #Quando criar o cronjob para correr este script diariamente, este for desaparece e day passa a ser 1 (yesterday)
    #for day in range(2, 4):
    day_txt = (date.today() - timedelta(days=2)).strftime('%Y%m%d')
        
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_daily_data'))   #check if folder MUR_dailyu_data exists in data folder
    if not exist_path:                                                            #if it don't exist:
        os.makedirs(os.path.join(base_path, 'data/MUR_daily_data'))               #create the folder

    download_sst(path = os.path.join(base_path, 'data/MUR_daily_data/'), date = pd.to_datetime(day_txt), mur_j0=12499, mur_j1=13499, mur_i0=16099, mur_i1=17499, replace=None)
    
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_algorithm_daily_images'))    #check if folder MUR_algorithm_daily_images exists in data folder
    if not exist_path:                                                                         #if doesn't exist
        os.makedirs(os.path.join(base_path, 'data/MUR_algorithm_daily_images'))                # create the folder
            
    
    df_yesterday_mur = get_data('sst_' + day_txt + '.nc', base_path=base_path)
    
    canny_front_detection_1day(df_yesterday_mur, day_txt, base_path=base_path)
    
    BOA_aplication(df_yesterday_mur, day_txt, base_path=base_path)
    
    CCA_front(df_yesterday_mur, day_txt, base_path=base_path)
        
    real_sst_image(df_yesterday_mur, day_txt, base_path=base_path)
    

if __name__ == "__main__":
    main()

