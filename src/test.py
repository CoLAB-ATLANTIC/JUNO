
############################################ download_mur_data_single_days.py  ##############################################
###                                                                                                                       ###
###    This script downloads MUR data individually, i.e.                                                                  ###
###    (it allows you to define a time period and then download the data for all the days in that period).                ###
###    These netCDF files are stored in folder: MUR_single_days                                                           ###
###                                                                                                                       ###
#############################################################################################################################

import os
import numpy as np
import pandas as pd
import wget
from math import floor
from pydap.client import open_url
from tqdm import tqdm
import sys
from datetime import date



##################################### FUNCTIONS TO DOWNLOAD MUR DATA ###########################################
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
    for date in tqdm(data_range, desc='SST', file=sys.stdout):
        download_sst(sst_path, date, mur_j0, mur_j1, mur_i0, mur_i1, replace)
    return

#######################################################################################################


def download_mur(base_path, years=10, from_start_date = '0601', to_end_date='0831'):
    
    """
    O objectivo desta função é fazer o download dos dados do MUR para um certo periodo (por exempolo no nosso caso queremos os Verões dos ultimos 10 anos, 
    portanto years=10, from_start_date='0601' (formato '%m%d' -> 1 de Junho) e to_end_date='0831' ->31 de Agosto. 
    merge_files_txt é o nome com que queremos que o ficheiro com os varios merged netCDFs fique guardado
    """

    start = []
    end = []

    #Create 2 list with the start and end of summer dates for the last 10 years (2011 to 2021)
    for i in range(1, years+1):        #range começa em 1 para ignorarmos 2022 e vai até years+2 (neste caso 2011)
        start.append(str((date.today().year-i)) + from_start_date)    # from_start_date='0601' -> começa a 1 de junho
        end.append(str((date.today().year-i)) + to_end_date)          # 'to_end_date = 0831' -> termina a 31 de Agosto

    for j in range(0, len(start)):
        download_sst_thread(data_range = pd.date_range(start=pd.to_datetime(start[j]), end=pd.to_datetime(end[j])), sst_path=os.path.join(base_path, 'data/MUR_last30days/'), mur_j0=12499, mur_j1=13499, mur_i0=16099, mur_i1=17499, replace=None)

        
        
def main():
    
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'JUNO')     
    
    exist_path = os.path.exists(os.path.join(base_path, 'data/MUR_last30days'))   #check if folder MUR_seasonal_data exists in data folder
    if not exist_path:                                                               #if it don't exist:
        os.makedirs(os.path.join(base_path, 'data/MUR_last30days'))               #create the folder
    
    
    #download_mur(base_path=base_path, years=2, from_start_date = '0601', to_end_date='0604')      
    download_mur(base_path=base_path, years=1, from_start_date = '0815', to_end_date='0913')     #download all data from every day from 2012 to 2021
     
    
    #FAZER DOWNLOAD DOS DIAS QUE FALTAM 2021-02-20 E 2021-02-21
    #download_mur(base_path=base_path, years=1, from_start_date = '0219', to_end_date='0222')  
    
    

if __name__ == "__main__":
    main()
       