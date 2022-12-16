#Import libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from shapely.geometry import Polygon
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import datetime
from bs4 import BeautifulSoup
import requests
import subprocess
import json
plt.rcParams["figure.figsize"] = 12, 10




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
        subprocess.run(['wget', '--user=' + eddies_user, '--password=' + eddies_pass, 'https://tds.aviso.altimetry.fr/thredds/fileServer/dataset-duacs-nrt-value-added-eddy-trajectory/'] + filename)
        os.chdir(base_path)    #return back to original directory
        
    exist_sliced_file = os.path.exists(path + filename[:-3] + '_slice.nc')
    if exist_sliced_file == False:   #smaller file with only the few variables we are interested in
        subprocess.run(['ncks', '-v', 'time,track,latitude,longitude,effective_contour_longitude,effective_contour_latitude', path + filename, path + filename[:-3] + '_slice.nc'])
    
    # after all I think it would be better to delete the bigger file
    #os.remove(path + filename)
    



######################## Agora vou fazer ainda outro slicing para termos do tempo e das coordenadas   ##############################################


def slice_netcdf(filepath, input_filename, lat_min, lat_max, lon_min, lon_max, date_init_str, date_end_str, output_filename):
    
    """
    With this function we want to take our sliced netcdf and extract a slice of only the variables we want
    Then we will merge everything together as a new, and much smaller netcdf file with the name output_filename in the AVISO folder
    path is the absolute path where our file is stored
    """
    
    date_init = datetime.datetime.strptime(date_init_str, '%Y-%m-%d').isoformat()
    date_init = np.datetime64(date_init)
    
    date_end = datetime.datetime.strptime(date_end_str, '%Y-%m-%d').isoformat()
    date_end = np.datetime64(date_end)
    
    file_path = os.path.join(filepath + input_filename)         #'test_slice.nc'
    data = xr.load_dataset(file_path)      # importar o netcdf como xarray


    #agora vou fazer um slice com base nos valores de latitude, longitude e time.
    #Isto vai gerar varios xarrays que depois vão ser merged e reconvertidos para um NetCDF que será substancialmente mais pequeno
    
    eddie_slice = (data['latitude'].values > lat_min) & (data['latitude'].values < lat_max) & (data['longitude'].values >lon_min) & (data['longitude'].values < lon_max) & (data['time'].values >= date_init) & (data['time'].values <= date_end)       #np.datetime64('2022-10-15T00:00:00')
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
    
    #Little arithmetic to get the proper values for the new xarray time variable attributes
    file_last_date_str = input_filename.split('_')[-2]
    file_last_date = datetime.datetime.strptime(file_last_date_str, '%Y%m%d').isoformat()
    file_last_date = np.datetime64(file_last_date)
    x = file_last_date - date_end
    diff_days = int(x / np.timedelta64(1, 'D'))
    y = date_end - date_init
    diff_days2 = int(y / np.timedelta64(1, 'D'))
    
    time.attrs['max'] = data['time'].attrs['max'] - diff_days 
    time.attrs['min'] = time.attrs['max'] - diff_days2
    

    tracking = data['track'][eddie_slice]
    tracking.attrs['min'] = tracking.values.min()
    tracking.attrs['max'] = tracking.values.max()


    att = data.attrs   #queremos que o novo xarray tenha os mesmos attributes que o dataset anterior
    test = xr.merge([lat, lon, effective_contour_lon, effective_contour_lat, time, tracking])
    test.attrs = att

    netcdf_save_folder = os.path.join(filepath + output_filename)
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
                
 
 
    # Create a json to store the 4 corners of the future image
    dictionary = {"llcorner": [35, -20],
                  "rlcorner": [35, -5],
                  "ulcorner": [45, -20],
                  "urcorner": [45, -5]}
    with open('/home/luisfigueiredo/JUNO/data/AVISO_images/json_bound_box', 'w') as json_file:
        json.dump(dictionary, json_file)
    
    
    return eddie_lats, eddie_lons, centro_y, centro_x

            

def eddies_visualization(eddie_cyc_lons, eddie_cyc_lats, centro_cyc_x, centro_cyc_y, eddie_anti_lons, eddie_anti_lats, centro_anti_x, centro_anti_y, title):
    
    """
    Function for the visualization of both eddies: cyclonic and anticyclonic
    """
    

    m = Basemap(projection='mill',
           llcrnrlat = 35,                      #35
           llcrnrlon = -20,                     #-20
           urcrnrlat = 45,                      # 45
           urcrnrlon = -5,                      #-5
           resolution = 'i')

    m.drawcoastlines()
    m.fillcontinents(color = 'darkgreen')   #put continental zones in darkgreen color

    #plot the shapes of the cyclonic eddies
    for i in range(len(eddie_cyc_lons)):
        x1_cyc, y1_cyc = m(eddie_cyc_lons[i], eddie_cyc_lats[i])
        m.plot(x1_cyc, y1_cyc, color='green')

    #all the centroid of the cyclonic eddies identified in a particular period
    for j in range(len(centro_cyc_x)):
        eddies_centro_cyc_x, eddies_centro_cyc_y = m(centro_cyc_x[j], centro_cyc_y[j])
        m.scatter(eddies_centro_cyc_x, eddies_centro_cyc_y, marker='.', color='m', linewidths=0.1)
        
    #plot the shapes of the anticyclonic eddies
    for i in range(len(eddie_anti_lons)):
        x1_anti, y1_anti = m(eddie_anti_lons[i], eddie_anti_lats[i])
        m.plot(x1_anti, y1_anti, color='red')

    #all the centroid of the cyclonic eddies identified in a particular period
    for j in range(len(centro_anti_x)):
        eddies_centro_anti_x, eddies_centro_anti_y = m(centro_anti_x[j], centro_anti_y[j])
        m.scatter(eddies_centro_anti_x, eddies_centro_anti_y, marker='.', color='m', linewidths=0.1)
        
    
    

    #m.drawparallels(np.arange(-90,90,2), labels = [True, False, False, False])
    #m.drawmeridians(np.arange(-180,180, 2), labels=[0,0,0,1])

    plt.title(title, fontsize=20)
    #plt.legend()
    #plt.show()
    plt.savefig('/home/luisfigueiredo/JUNO/data/AVISO_images/' + title)
 

def main():
    
    
    #################### Sacar conteudo html da pagina para obter o nome dos ficheiros que quero fazer download ###########################

    resposta = requests.get('https://tds.aviso.altimetry.fr/thredds/catalog/dataset-duacs-nrt-value-added-eddy-trajectory/catalog.html')
    site = BeautifulSoup(resposta.content, 'html.parser')
    a = site.find_all('a')
    t = []
    for link in a:
        tt = link.find('tt')
        t.append(tt)
        
    cyclonic_name = t[1].text        #nome do ficheiro NETcdf para os dados ciclonicos
    anticyclonic_name = t[2].text    #nome do ficheiro NETcdf para os dados anticiclonicos

    eddies_user = os.environ.get('EDDIES_USER')
    eddies_pass = os.environ.get('EDDIES_PASS')
    
    #fazer o download dos ficheiros cujos filenames foram identificados através do request feito no conteudo HTML do site do AVISO
    download_eddie(filename=cyclonic_name, path = '/home/luisfigueiredo/JUNO/data/AVISO_data/', eddies_user=eddies_user, eddies_pass = eddies_pass)
    download_eddie(filename=anticyclonic_name, path = '/home/luisfigueiredo/JUNO/data/AVISO_data/', eddies_user=eddies_user, eddies_pass = eddies_pass)
    
    #last slicing according to certain variable boundaries (coordinates and time)
    slice_netcdf(filepath = '/home/luisfigueiredo/JUNO/data/AVISO_data/', input_filename=cyclonic_name[:-3] + '_slice.nc', lat_min=35, lat_max=45, lon_min=330, lon_max=360, date_init_str='2022-10-15', date_end_str='2022-10-23', output_filename='cyclonic_test.nc')
    slice_netcdf(filepath = '/home/luisfigueiredo/JUNO/data/AVISO_data/', input_filename=anticyclonic_name[:-3] + '_slice.nc', lat_min=35, lat_max=45, lon_min=330, lon_max=360, date_init_str='2022-10-15', date_end_str='2022-10-23', output_filename='anticyclonic_test.nc')

    #eddies tracking and identification for the cyclonic and anticyclonic data
    eddie_cyc_lats, eddie_cyc_lons, centro_cyc_y, centro_cyc_x = eddie_tracking(filepath='/home/luisfigueiredo/JUNO/data/AVISO_data/', filename='cyclonic_test.nc')
    eddie_anti_lats, eddie_anti_lons, centro_anti_y, centro_anti_x = eddie_tracking(filepath='/home/luisfigueiredo/JUNO/data/AVISO_data/', filename='anticyclonic_test.nc')
    
    #visualize the eddies. (Then we can maybe save the plot as a png)
    eddies_visualization(eddie_cyc_lons, eddie_cyc_lats, centro_cyc_x, centro_cyc_y, eddie_anti_lons, eddie_anti_lats, centro_anti_x,
                     centro_anti_y, title = 'Eddies from 15-10-2022 to 23-10-2022')


if __name__ == '__main__':
    main()