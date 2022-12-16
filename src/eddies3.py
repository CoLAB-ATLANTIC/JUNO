#Import libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from bs4 import BeautifulSoup
import requests
import subprocess
import json
plt.rcParams["figure.figsize"] = 12, 10
plt.rcParams["figure.autolayout"] = True
import cv2
import netCDF4 as nc




################################ Usar subprocess para usar o python como se estivesse na command line ###############################################
######################################### Download the eddies file exactly how i wanted it ##########################################################

def download_eddie(filename, path, eddies_user, eddies_pass):
    
    """
    Function to download the data from the AVISO (eddies) (if its not already downloaded). Requires the filename (name of cyclonic or anticyclonic file)
    we want to give our data, the absolute path where it will be stored and a username and password from the AVISO to be able to download eddies data.
    Then we want to get a smaller file with only the variables we are interested in (time, track, lon, lat, effective_contour_lat, effective_contour_lon)
    """

    base_path = os.getcwd()
    exist_eddies_file = os.path.exists(path + filename)       
    if exist_eddies_file == False:       #if the file does not exist in that folder
        os.chdir(path)   #change directory to where we want to download and use subprocess to apply terminal commands directly in Python (to download the data)
        subprocess.run(['wget', '--user=' + eddies_user, '--password=' + eddies_pass, 'https://tds.aviso.altimetry.fr/thredds/fileServer/dataset-duacs-nrt-value-added-eddy-trajectory/' + filename])
        os.chdir(base_path)    #return back to original directory
        
    exist_sliced_file = os.path.exists(path + filename[:-3] + '_slice.nc')
    if exist_sliced_file == False:   #if the smaller file with only the few variables we are interested in does not exist
        subprocess.run(['ncks', '-v', 'time,track,latitude,longitude,effective_contour_longitude,effective_contour_latitude', path + filename, path + filename[:-3] + '_slice.nc'])
    
    # after all I think it would be better to delete the bigger file
    #os.remove(path + filename)
    
    
    
######################## Agora vou fazer ainda outro slicing para termos do tempo e das coordenadas   ##############################################


def slice_netcdf(filepath, input_filename, lat_min, lat_max, lon_min, lon_max, data_final_str, eddie_type):
    
    """
    With this function we want to take our sliced netcdf and extract a slice of only the variables we want
    Then we will merge everything together as a new, and much smaller netcdf file with the name output_filename in the AVISO folder
    path is the absolute path where our file is stored
    """
    
    file_last_date = datetime.datetime.strptime(data_final_str, '%Y%m%d').isoformat()
    #file_last_date = data_final_str.strptime('%Y%m%d')
    #file_last_date = data_final_str.isoformat()
    file_last_date = np.datetime64(file_last_date)
    
    
    #we want information regarding the last 7 days of the netcdf
    date_init = file_last_date - np.timedelta64(7,'D')
    
    file_path = os.path.join(filepath + input_filename)         #'test_slice.nc'
    data = xr.load_dataset(file_path)      # importar o netcdf como xarray


    #agora vou fazer um slice com base nos valores de latitude, longitude e time.
    #Isto vai gerar varios xarrays que depois vão ser merged e reconvertidos para um NetCDF que será substancialmente mais pequeno
    
    eddie_slice = (data['latitude'].values > lat_min) & (data['latitude'].values < lat_max) & (data['longitude'].values >lon_min) & (data['longitude'].values < lon_max) & (data['time'].values >= date_init) & (data['time'].values <= file_last_date)       #np.datetime64('2022-10-15T00:00:00')
    lat = data['latitude'][eddie_slice]       #np.datetime64('2022-10-15T00:00:00')
    lat.attrs['min'] = lat.values.min()
    lat.attrs['max'] = lat.values.max()

    lon = data['longitude'][eddie_slice]
    lon.attrs['min'] = lon.values.min()
    lon.attrs['max'] = lon.values.max()

    effective_contour_lon = data['effective_contour_longitude'][eddie_slice]
    effective_contour_lon.attrs['min'] = effective_contour_lon.values.min()
    effective_contour_lon.attrs['max'] = effective_contour_lon.values.max()

    effective_contour_lat = data['effective_contour_latitude'][eddie_slice]
    effective_contour_lat.attrs['min'] = effective_contour_lat.values.min()
    effective_contour_lat.attrs['max'] = effective_contour_lat.values.max()

    #Isto depois vai ter de ser adaptado conforme as datas que queiramos. Neste caso esta de dia 15 a 23 (8 dias)
    time = data['time'][eddie_slice]
        
    time.attrs['max'] = data['time'].attrs['max'] 
    time.attrs['min'] = time.attrs['max'] - 7
    

    tracking = data['track'][eddie_slice]
    tracking.attrs['min'] = tracking.values.min()
    tracking.attrs['max'] = tracking.values.max()


    att = data.attrs   #queremos que o novo xarray tenha os mesmos attributes que o dataset anterior
    test = xr.merge([lat, lon, effective_contour_lon, effective_contour_lat, time, tracking])
    test.attrs = att

    netcdf_save_folder = os.path.join(filepath + eddie_type + '_' + data_final_str + '.nc')
    test.to_netcdf(path=netcdf_save_folder)
    
    #I think that after this final sliced we can delete the previous netCDF (for space, memory and cleaning purposes)
    #os.remove(os.path.join(filepath + cyclonic_name[:-3] + 'slice.nc'))
    #os.remove(os.path.join(filepath + anticyclonic_name[:-3] + 'slice.nc'))





############################### Agora vou importar o NetCDF que acabou de ser criado como um xarray ########################################



def eddie_tracking(filepath, filename):
    
    """
    Function that receives the file with the eddies in a NetCDF format. 
    It returns lists with the coordinates of the shape of the eddies and the coordinates of the centroids of those eddies.
    For each eddie might be several centroids since that eddie might be identified several times for the netcdf timeframe 
    """
    
    netcdf_path = os.path.join(filepath + filename)
    data_xr = xr.load_dataset(netcdf_path)

    track = data_xr['track'].values

    eddies_track = np.unique(track)

    track_list = []
    track = list(track)
    # Eddies identified in the location: lat[35; 45] and lon[330;350] from 20 of August to 30 of August 2022
    eddies_track = np.unique(track)
    for value in eddies_track:
        x = track.count(value)
        track_list.append([value, x])
    
    eddie_lons = []
    eddie_lats = []

    centro_x = []
    centro_y = []
    for value in eddies_track:


        #-360 nos valores maiores que 180
        x = data_xr['effective_contour_longitude'].values[(data_xr['track'] == value)][-1] - 360
       # for i in range(len(x)):
        #    if x[i] > 180:
         #       x[i] = x[i]-360
        y = data_xr['effective_contour_latitude'].values[(data_xr['track'] == value)][-1] 
    
        eddie_lons.append(list(x))
        eddie_lats.append(list(y))
    
    #find center coordinates of all the eddies identified in those coordinates in that period
    for value in track_list:
    
        for i in range(1, value[1] + 1):
    
            eddie_x = data_xr['effective_contour_longitude'].values[(data_xr['track'] == value[0])][-i] - 360
            eddie_y = data_xr['effective_contour_latitude'].values[(data_xr['track'] == value[0])][-i] 

            poly = Polygon(zip(eddie_x,eddie_y))

            poly_centroid_x = poly.centroid.coords[0][0]
            poly_centroid_y = poly.centroid.coords[0][1]
    
            if poly_centroid_x != -180.0:
                centro_x.append(poly_centroid_x)  #coordinates of the centroid
            if poly_centroid_y != 0.0:
                centro_y.append(poly_centroid_y)
    
    
    return eddie_lats, eddie_lons, centro_y, centro_x



def eddies_arrays(eddie_cyc_lons, eddie_cyc_lats, centro_cyc_x, centro_cyc_y, eddie_anti_lons, eddie_anti_lats, centro_anti_x, centro_anti_y, file_last_date_str):
    
    """
    Function to save 4 images as .png files: the shapes of the eddies (cyclonic and anticyclonic) and the respective centroids.
    Then it loads the 4 images and converts them to a numpy array 
    """
    

    #plot the shapes of the cyclonic eddies
    fig = plt.figure(figsize=(12,10))
    for i in range(len(eddie_cyc_lons)):
        plt.plot(eddie_cyc_lons[i], eddie_cyc_lats[i], color='green')
    plt.xlim([-20, -4]);
    plt.ylim([33, 46]);
    plt.axis('off');
    plt.savefig('/home/luisfigueiredo/JUNO/data/AVISO_images/shape_cyc_' + file_last_date_str + '.png')
    
    #array with cyclonic shape data
    im_cyc = cv2.imread('/home/luisfigueiredo/JUNO/data/AVISO_images/shape_cyc_' + file_last_date_str + '.png')
    array_cyc_shape = cv2.cvtColor(im_cyc, cv2.COLOR_BGR2GRAY).astype('float32')    # BGR -> GRAYSCALE
    array_cyc_shape[array_cyc_shape<255] = 0
    array_cyc_shape[array_cyc_shape==255] = np.nan
    array_cyc_shape[array_cyc_shape==0] = 1
    array_cyc_shape = np.flipud(array_cyc_shape)


    #all the centroid of the cyclonic eddies identified in a particular period
    fig = plt.figure(figsize=(12,10))
    for j in range(len(centro_cyc_x)):
        plt.scatter(centro_cyc_x[j], centro_cyc_y[j], marker='.', color='m', linewidths=0.1)
    plt.xlim([-20, -4]);
    plt.ylim([33, 46]);
    plt.axis('off');
    plt.savefig('/home/luisfigueiredo/JUNO/data/AVISO_images/centers_cyc_' + file_last_date_str + '.png')
    #array with cyclonic centroids data
    im_cyc_centroids = cv2.imread('/home/luisfigueiredo/JUNO/data/AVISO_images/centers_cyc_' + file_last_date_str + '.png')
    array_cyc_centroids = cv2.cvtColor(im_cyc_centroids, cv2.COLOR_BGR2GRAY).astype('float32')   # BGR -> GRAYSCALE
    array_cyc_centroids[array_cyc_centroids<255] = 0
    array_cyc_centroids[array_cyc_centroids==255] = np.nan
    array_cyc_centroids[array_cyc_centroids==0] = 1
    array_cyc_centroids = np.flipud(array_cyc_centroids)
        
    #save image of the shapes of the anticyclonic eddies
    fig = plt.figure(figsize=(12,10))
    for i in range(len(eddie_anti_lons)):
        plt.plot(eddie_anti_lons[i], eddie_anti_lats[i], color='red')
    plt.xlim([-20, -4]);
    plt.ylim([33, 46]);
    plt.axis('off');
    plt.savefig('/home/luisfigueiredo/JUNO/data/AVISO_images/shape_anti_' + file_last_date_str + '.png')
    #array with anticyclonic shape data
    im_anti = cv2.imread('/home/luisfigueiredo/JUNO/data/AVISO_images/shape_anti_' + file_last_date_str + '.png')
    array_anti_shape = cv2.cvtColor(im_anti, cv2.COLOR_BGR2GRAY).astype('float32')    # BGR -> GRAYSCALE
    array_anti_shape[array_anti_shape<255] = 0
    array_anti_shape[array_anti_shape==255] = np.nan
    array_anti_shape[array_anti_shape==0] = 1
    array_anti_shape = np.flipud(array_anti_shape)

    #all the centroid of the cyclonic eddies identified in a particular period
    fig = plt.figure(figsize=(12,10))
    for j in range(len(centro_anti_x)):
        plt.scatter(centro_anti_x[j], centro_anti_y[j], marker='.', color='m', linewidths=0.1)
    plt.xlim([-20, -4]);
    plt.ylim([33, 46]);
    plt.axis('off');
    plt.savefig('/home/luisfigueiredo/JUNO/data/AVISO_images/centers_anti_' + file_last_date_str + '.png')
    #array with anticyclonic centroids data
    im_anti_centroids = cv2.imread('/home/luisfigueiredo/JUNO/data/AVISO_images/centers_anti_' + file_last_date_str + '.png')
    array_anti_centroids = cv2.cvtColor(im_anti_centroids, cv2.COLOR_BGR2GRAY).astype('float32') # BGR -> GRAYSCALE
    array_anti_centroids[array_anti_centroids<255] = 0
    array_anti_centroids[array_anti_centroids==255] = np.nan
    array_anti_centroids[array_anti_centroids==0] = 1
    array_anti_centroids = np.flipud(array_anti_centroids)
    
    return array_cyc_shape, array_cyc_centroids, array_anti_shape, array_anti_centroids



def request_eddy_filenames():
    
    resposta = requests.get('https://tds.aviso.altimetry.fr/thredds/catalog/dataset-duacs-nrt-value-added-eddy-trajectory/catalog.html')
    site = BeautifulSoup(resposta.content, 'html.parser')
    a = site.find_all('a')
    t = []
    for link in a:
        tt = link.find('tt')
        t.append(tt)
        
    cyclonic_name = t[1].text        #nome do ficheiro NETcdf para os dados ciclonicos
    anticyclonic_name = t[2].text    #nome do ficheiro NETcdf para os dados anticiclonicos
    
    return cyclonic_name, anticyclonic_name



def create_netcdf(data_final_str, array_cyc_shape, array_cyc_centroids, array_anti_shape, array_anti_centroids):
    
    nc_file = '/home/luisfigueiredo/JUNO/data/AVISO_netcdf/eddies_' + data_final_str + '.nc'

    if os.path.exists(nc_file):
        os.remove(nc_file)

    ds = nc.Dataset(nc_file, 'w', format='NETCDF4')

    ds.title = 'Eddies Netcdf ' + data_final_str

    height = array_cyc_shape.shape[0]
    width = array_cyc_shape.shape[1]

    #create dimensions of the NetCDF file
    #time = ds.createDimension('time')
    lat = ds.createDimension('lat', height)
    lon = ds.createDimension('lon', width)

    #times = ds.createVariable('time', 'f4', ('time', ))
    lats = ds.createVariable('lat', 'f4', ('lat', ))
    lons = ds.createVariable('lon', 'f4', ('lon', ))


    cyclonic_shape = ds.createVariable('cyc_shape', 'i8', ('lat', 'lon',)) 
    cyclonic_shape.units = 'X'
    cyclonic_shape.description = 'Array with the shapes of the cyclonic eddies for that day'
    cyclonic_shape[:, :] = array_cyc_shape

    cyclonic_centroids = ds.createVariable('cyc_centr', 'i8', ('lat', 'lon',)) 
    cyclonic_centroids.units = 'Y'
    cyclonic_centroids.description = 'Array with the dots with the centroids of the identified cyclonic eddies for a week of movement'
    cyclonic_centroids[:, :] = array_cyc_centroids


    anticyclonic_shape = ds.createVariable('anticyc_shape', 'i8', ('lat', 'lon',)) 
    anticyclonic_shape.units = 'W'
    anticyclonic_shape.description = 'Array with the shapes of the anticyclonic eddies for that day'
    anticyclonic_shape[:, :] = array_anti_shape

    anticyclonic_centroids = ds.createVariable('anticyc_centr', 'i8', ('lat', 'lon',)) 
    anticyclonic_centroids.units = 'Z'
    anticyclonic_centroids.description = 'Array with the dots with the centroids of the identified anticyclonic eddies for a week of movement'
    anticyclonic_centroids[:, :] = array_anti_centroids

    lats[:] = np.linspace(33, 46, height)
    lons[:] = np.linspace(-20, -4, width)

    ds.close() 
    
    
############################################################# MAIN FUNCTION ################################################################


def main():
    

    # Check if there are files in the folder AVISO_data. If so the files in that folder they will have the following format:
    #  cyclonic_date.nc                             OR                        anticyclonic_date.nc
    if os.listdir('/home/luisfigueiredo/JUNO/data/AVISO_data') == []:       #if the folder AVISO_data is empty
        
        #get the html content of the page to get the name of the files we wish to download
        cyclonic_name, anticyclonic_name = request_eddy_filenames()
        
        #get the most recent date for which the AVISO eddies netcdf has data
        x = cyclonic_name.split('_')[-1]
        last_date_str = x.split('.')[0]

        #get eddies credentials (username and password) that are hidden as environment variables
        eddies_user = "luis.leao.figueiredo.23@gmail.com"
        eddies_pass = "17PLqM"
        #eddies_user = os.environ.get('EDDIES_USER')
        #eddies_pass = os.environ.get('EDDIES_PASS')
        
        # download the 2 files which filenames were identified through the request made in the HTML content of the AVISO website
        download_eddie(filename=cyclonic_name, path = '/home/luisfigueiredo/JUNO/data/AVISO_data/', eddies_user=eddies_user, eddies_pass = eddies_pass)
        download_eddie(filename=anticyclonic_name, path = '/home/luisfigueiredo/JUNO/data/AVISO_data/', eddies_user=eddies_user, eddies_pass = eddies_pass)
        
        #last slicing according to certain variable boundaries (coordinates and time)
        #this will result in 2 much smaller netcdfs (cyclonic and anticyclonic)
        slice_netcdf(filepath = '/home/luisfigueiredo/JUNO/data/AVISO_data/', input_filename=cyclonic_name[:-3] + '_slice.nc', lat_min=35, lat_max=45, lon_min=330, lon_max=360, data_final_str=last_date_str, eddie_type='cyclonic')
        slice_netcdf(filepath = '/home/luisfigueiredo/JUNO/data/AVISO_data/', input_filename=anticyclonic_name[:-3] + '_slice.nc', lat_min=35, lat_max=45, lon_min=330, lon_max=360, data_final_str=last_date_str, eddie_type='anticyclonic')

        #I think that after this final sliced we can delete the bigger netCDF (for space, memory and cleaning purposes)
        os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ cyclonic_name)
        os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ anticyclonic_name)
        os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ cyclonic_name[:-3] + '_slice.nc')
        os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ anticyclonic_name[:-3] + '_slice.nc')
            
        #eddies tracking and identification for the cyclonic and anticyclonic data
        eddie_cyc_lats, eddie_cyc_lons, centro_cyc_y, centro_cyc_x = eddie_tracking(filepath='/home/luisfigueiredo/JUNO/data/AVISO_data/', filename='cyclonic_' + last_date_str + '.nc')
        eddie_anti_lats, eddie_anti_lons, centro_anti_y, centro_anti_x = eddie_tracking(filepath='/home/luisfigueiredo/JUNO/data/AVISO_data/', filename='anticyclonic_' + last_date_str + '.nc')
        
        #visualize the eddies. (Then we can maybe save the plot as a png)
        array_cyc_shape, array_cyc_centroids, array_anti_shape, array_anti_centroids = eddies_arrays(eddie_cyc_lons, eddie_cyc_lats, centro_cyc_x, centro_cyc_y, 
                                                                    eddie_anti_lons, eddie_anti_lats, centro_anti_x,centro_anti_y, file_last_date_str = last_date_str)


        exist_path = os.path.exists('/home/luisfigueiredo/JUNO/data/AVISO_netcdf')    #check if folder AVISO_netcdf exists in data folder
        if not exist_path:                                                                         #if doesn't exist
            os.makedirs('/home/luisfigueiredo/JUNO/data/AVISO_netcdf')                   # create the folder
            
            
        # CREATION OF THE NETCDF   
        create_netcdf(data_final_str=last_date_str, array_cyc_shape = array_cyc_shape, array_cyc_centroids=array_cyc_centroids,
                      array_anti_shape = array_anti_shape, array_anti_centroids = array_anti_centroids)
        
               
        
        
    else:    #if the folder AVISO_data is not empty
        #we want to extract the most recent date for the files in the directory
        list_dates = [] 
        for filename in os.listdir('/home/luisfigueiredo/JUNO/data/AVISO_data'):
            x  = filename.split('_')[-1]
            data = x.split('.')[0]
            list_dates.append(data)
        recent_date_str = max(list_dates)      #most recent data for the files in the AVISO_data folder
        
    
        # get the html content of the page to get the name of the files we wish to download
        cyclonic_name, anticyclonic_name = request_eddy_filenames()
    
        #get the most recent date for the AVISO eddies netcdf that we just requested
        x = cyclonic_name.split('_')[-1]
        last_date_str = x.split('.')[0]
    
        #convert the dates from string to datetime object
        date_recent = datetime.datetime.strptime(recent_date_str, '%Y%m%d')   #data relativa ao ficheiro mais recente presente na folder AVISO_data
        last_date = datetime.datetime.strptime(last_date_str, '%Y%m%d')       #ultima data disponivel para os dados do AVISO
    
        #if the last_date (of the AVISO file requested from the web) is the same as the date we have from the most recent file in our AVISO_data directory -> pass
        if last_date == date_recent:      
            pass
        elif last_date > date_recent:        
            #get eddies credentials (username and password) that are hidden as environment variables
            eddies_user = os.environ.get('EDDIES_USER')
            eddies_pass = os.environ.get('EDDIES_PASS')
        
            #make the donwload of the 2 files (cyclonic and anticyclonic) that were identified through the request in the HTML of the AVISO website
            download_eddie(filename=cyclonic_name, path = '/home/luisfigueiredo/JUNO/data/AVISO_data/', eddies_user=eddies_user, eddies_pass = eddies_pass)
            download_eddie(filename=anticyclonic_name, path = '/home/luisfigueiredo/JUNO/data/AVISO_data/', eddies_user=eddies_user, eddies_pass = eddies_pass)
        
            #we can remove the bigger netcdf files
            os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ cyclonic_name)
            os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ anticyclonic_name)
        
            #Now we want
            while last_date > date_recent:
            
                data_final = last_date
                data_final_str = data_final.strftime('%Y%m%d')

                #last slicing according to certain variable boundaries (coordinates and time)
                slice_netcdf(filepath = '/home/luisfigueiredo/JUNO/data/AVISO_data/', input_filename=cyclonic_name[:-3] + '_slice.nc', lat_min=35, lat_max=45, lon_min=330, lon_max=360, data_final_str=data_final, eddie_type='cyclonic')
                slice_netcdf(filepath = '/home/luisfigueiredo/JUNO/data/AVISO_data/', input_filename=anticyclonic_name[:-3] + '_slice.nc', lat_min=35, lat_max=45, lon_min=330, lon_max=360, data_final_str=data_final, eddie_type='anticyclonic')
            
                #eddies tracking and identification for the cyclonic and anticyclonic data
                eddie_cyc_lats, eddie_cyc_lons, centro_cyc_y, centro_cyc_x = eddie_tracking(filepath='/home/luisfigueiredo/JUNO/data/AVISO_data/', filename='cyclonic_' + data_final_str + '.nc')
                eddie_anti_lats, eddie_anti_lons, centro_anti_y, centro_anti_x = eddie_tracking(filepath='/home/luisfigueiredo/JUNO/data/AVISO_data/', filename='anticyclonic_' + data_final_str + '.nc')
    
                #visualize the eddies. (Then we can maybe save the plot as a png)
                array_cyc_shape, array_cyc_centroids, array_anti_shape, array_anti_centroids = eddies_arrays(eddie_cyc_lons, eddie_cyc_lats, centro_cyc_x, centro_cyc_y, 
                                                                eddie_anti_lons, eddie_anti_lats, centro_anti_x,centro_anti_y, file_last_date = data_final_str)


                exist_path = os.path.exists('/home/luisfigueiredo/JUNO/data/AVISO_netcdf')    #check if folder AVISO_netcdf exists in data folder
                if not exist_path:                                                                         #if doesn't exist
                    os.makedirs('/home/luisfigueiredo/JUNO/data/AVISO_netcdf')                   # create the folder
        
        
                #  CREATION OF THE NETCDF   
                create_netcdf(data_final_str=data_final_str, array_cyc_shape = array_cyc_shape, array_cyc_centroids=array_cyc_centroids,
                      array_anti_shape = array_anti_shape, array_anti_centroids = array_anti_centroids)


                last_date = last_date - timedelta(days=1)
            
            #in the end we can remove the sliced netcdfs so basically we will only have the tiny netcdfs with those certain boundaries
            os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ cyclonic_name[:-3] + '_slice.nc')
            os.remove('/home/luisfigueiredo/JUNO/data/AVISO_data/'+ anticyclonic_name[:-3] + '_slice.nc')
        


if __name__ == '__main__':
    main()
    
    

        
