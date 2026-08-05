"""
AVISO Mesoscale Eddy Daily Processing Script

This script downloads the latest near real time cyclonic and anticyclonic eddy trajectory datasets from the AVISO THREDDS server and generates daily
NetCDF products for the region around the coast of Portugal .

The main processing steps are:

1. Read the AVISO credentials from the EDDIES_USER and EDDIES_PASS environment variables.
2. Retrieve the latest cyclonic and anticyclonic NetCDF filenames from the AVISO THREDDS XML catalogue.
3. Identify missing daily outputs by comparing the dates available from AVISO with the existing files named eddies_YYYYMMDD.nc.
4. Download and reduce the AVISO datasets to the required variables.
5. Extract eddies within the configured geographic and temporal boundaries.
6. Identify the contours and centres of cyclonic and anticyclonic eddies.
7. Render the eddy shapes and centroids in memory and generate daily NetCDF files.
8. Remove intermediate files after successful processing to reduce storage usage.

The processing is resumable so after an interruption the script processes only the dates that are still missing.
"""



#Import libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from shapely.geometry import Polygon
import datetime
from datetime import timedelta
from bs4 import BeautifulSoup
import requests
import subprocess
import json
import re
import xml.etree.ElementTree as ET
plt.rcParams["figure.figsize"] = 12, 10
plt.rcParams["figure.autolayout"] = True
import netCDF4 as nc




################################ Usar subprocess para usar o python como se estivesse na command line ###############################################
######################################### Download the eddies file exactly how i wanted it ##########################################################


def download_eddie(filename, path):
    """
    Download an AVISO eddy NetCDF file and create a smaller NetCDF containing
    only the required variables.

    Credentials are read from the EDDIES_USER and EDDIES_PASS environment
    variables.
    """

    eddies_user = os.environ.get("EDDIES_USER")
    eddies_pass = os.environ.get("EDDIES_PASS")

    if not eddies_user or not eddies_pass:
        raise RuntimeError(
            "EDDIES_USER and EDDIES_PASS environment variables are not defined"
        )

    os.makedirs(path, exist_ok=True)

    download_url = ("https://tds-odatis.aviso.altimetry.fr/thredds/fileServer/"
                    "dataset-duacs-nrt-value-added-eddy-trajectory/"
                    f"{filename}")

    output_file = os.path.join(path, filename)
    temporary_file = output_file + ".part"

    sliced_file = os.path.join(path, f"{os.path.splitext(filename)[0]}_slice.nc")

    # Download the original file if it does not already exist
    if not os.path.exists(output_file):
        print(f"Downloading: {download_url}")
        print(f"Destination: {output_file}")

        try:
            with requests.get(download_url, auth=(eddies_user, eddies_pass), stream=True, timeout=(30, 600)) as response:
                response.raise_for_status()

                with open(temporary_file, "wb") as file:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            file.write(chunk)

            # Rename only after a successful download
            os.replace(temporary_file, output_file)

            print(f"Download completed: {output_file}")

        except requests.RequestException as error:
            if os.path.exists(temporary_file):
                os.remove(temporary_file)

            raise RuntimeError(f"Failed to download {filename} from AVISO: {error}") from error

    else:
        print(f"File already exists: {output_file}")

    # Make sure the downloaded file exists and is not empty
    if not os.path.isfile(output_file):
        raise FileNotFoundError(f"The downloaded file was not found: {output_file}")

    if os.path.getsize(output_file) == 0:
        raise RuntimeError(f"The downloaded file is empty: {output_file}")

    # Create the reduced NetCDF if it does not already exist
    if not os.path.exists(sliced_file):
        print(f"Creating reduced NetCDF: {sliced_file}")

        variables = (
            "time,track,latitude,longitude,"
            "effective_contour_longitude,"
            "effective_contour_latitude"
        )

        try:
            subprocess.run(
                [
                    "ncks",
                    "-O",
                    "-v",
                    variables,
                    output_file,
                    sliced_file,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

        except FileNotFoundError as error:
            raise RuntimeError(
                "The 'ncks' command was not found. "
                "Make sure NCO is installed and available to the cronjob."
            ) from error

        except subprocess.CalledProcessError as error:
            if os.path.exists(sliced_file):
                os.remove(sliced_file)

            raise RuntimeError(f"ncks failed while processing {output_file}:\n"  f"{error.stderr}") from error

        print(f"Reduced NetCDF created: {sliced_file}")

    else:
        print(f"Reduced NetCDF already exists: {sliced_file}")
    
    
    
######################## Agora vou fazer ainda outro slicing para termos do tempo e das coordenadas   ##############################################


def slice_netcdf(filepath, input_filename, lat_min, lat_max, lon_min, lon_max, data_final_str, eddie_type):
    
    """
    With this function we want to take our sliced netcdf and extract a slice of only the variables we want
    Then we will merge everything together as a new, and much smaller netcdf file in the AVISO folder
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
    # Filtering integer variables introduces NaNs. Clear the source encodings
    # so xarray does not try to serialize those NaNs back into integer dtypes.
    for variable_name in test.variables:
        test[variable_name].encoding = {}

    test.to_netcdf(path=netcdf_save_folder)
    





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



def _figure_to_binary_array(fig):
    """Convert a rendered Matplotlib figure to the existing 1/NaN mask."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())

    # Any non-white rendered pixel represents an eddy line or centroid.
    rendered_feature = np.any(rgba[:, :, :3] < 255, axis=2)
    result = np.full(rendered_feature.shape, np.nan, dtype=np.float32)
    result[rendered_feature] = 1.0

    # Preserve the orientation used by the former PNG/OpenCV workflow.
    return np.flipud(result)


def _new_eddy_figure():
    """Create a figure with the same size and geographic limits as before."""
    fig, axis = plt.subplots(figsize=(12, 10))
    axis.set_xlim(-20, -4)
    axis.set_ylim(33, 46)
    axis.axis("off")
    return fig, axis


def eddies_arrays(
    eddie_cyc_lons,
    eddie_cyc_lats,
    centro_cyc_x,
    centro_cyc_y,
    eddie_anti_lons,
    eddie_anti_lats,
    centro_anti_x,
    centro_anti_y,
):
    """
    Render cyclonic and anticyclonic shapes and centroids directly in memory.

    This preserves the four arrays used by create_netcdf without creating PNG
    files in the AVISO_images directory.
    """
    fig, axis = _new_eddy_figure()
    for longitudes, latitudes in zip(eddie_cyc_lons, eddie_cyc_lats):
        axis.plot(longitudes, latitudes, color="green")
    array_cyc_shape = _figure_to_binary_array(fig)
    plt.close(fig)

    fig, axis = _new_eddy_figure()
    for longitude, latitude in zip(centro_cyc_x, centro_cyc_y):
        axis.scatter(
            longitude,
            latitude,
            marker=".",
            color="m",
            linewidths=0.1,
        )
    array_cyc_centroids = _figure_to_binary_array(fig)
    plt.close(fig)

    fig, axis = _new_eddy_figure()
    for longitudes, latitudes in zip(eddie_anti_lons, eddie_anti_lats):
        axis.plot(longitudes, latitudes, color="red")
    array_anti_shape = _figure_to_binary_array(fig)
    plt.close(fig)

    fig, axis = _new_eddy_figure()
    for longitude, latitude in zip(centro_anti_x, centro_anti_y):
        axis.scatter(
            longitude,
            latitude,
            marker=".",
            color="m",
            linewidths=0.1,
        )
    array_anti_centroids = _figure_to_binary_array(fig)
    plt.close(fig)

    return (
        array_cyc_shape,
        array_cyc_centroids,
        array_anti_shape,
        array_anti_centroids,
    )



AVISO_HOST = "https://tds-odatis.aviso.altimetry.fr"

CATALOG_URL = (
    f"{AVISO_HOST}/thredds/catalog/"
    "dataset-duacs-nrt-value-added-eddy-trajectory/catalog.xml"
)


def request_eddy_filenames(eddies_user, eddies_pass):
    """Return the newest cyclonic and anticyclonic AVISO filenames."""
    try:
        response = requests.get(
            CATALOG_URL,
            auth=(eddies_user, eddies_pass),
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(f"Could not retrieve the AVISO catalogue: {error}") from error

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as error:
        raise RuntimeError("AVISO did not return a valid THREDDS XML catalogue. " f"Response preview: {response.text[:500]}") from error

    filenames = []
    for dataset in root.findall(".//{*}dataset"):
        name = dataset.get("name", "").strip()
        if name.lower().endswith(".nc"):
            filenames.append(name)

    filenames = list(dict.fromkeys(filenames))
    cyclonic_files = [
        name for name in filenames
        if "cyclonic" in name.lower()
        and "anticyclonic" not in name.lower()
    ]
    anticyclonic_files = [
        name for name in filenames
        if "anticyclonic" in name.lower()
    ]

    if not cyclonic_files or not anticyclonic_files:
        raise RuntimeError("Could not identify both eddy files in the AVISO catalogue. " f"NetCDF files found: {filenames}")

    cyclonic_name = max(cyclonic_files)
    anticyclonic_name = max(anticyclonic_files)
    print(f"Cyclonic file found: {cyclonic_name}", flush=True)
    print(f"Anticyclonic file found: {anticyclonic_name}", flush=True)
    return cyclonic_name, anticyclonic_name



def create_netcdf(data_final_str, array_cyc_shape, array_cyc_centroids, array_anti_shape, array_anti_centroids):
    
    nc_file = '/home/colabatlantic2/projects/JUNO/data/AVISO_netcdf/eddies_' + data_final_str + '.nc'

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


    fill_value = np.int8(-127)

    cyclonic_shape = ds.createVariable('cyc_shape', 'i1', ('lat', 'lon',), fill_value=fill_value)
    cyclonic_shape.units = 'X'
    cyclonic_shape.description = 'Array with the shapes of the cyclonic eddies for that day'
    cyclonic_shape[:, :] = np.ma.masked_invalid(array_cyc_shape)

    cyclonic_centroids = ds.createVariable('cyc_centr', 'i1', ('lat', 'lon',), fill_value=fill_value)
    cyclonic_centroids.units = 'Y'
    cyclonic_centroids.description = 'Array with the dots with the centroids of the identified cyclonic eddies for a week of movement'
    cyclonic_centroids[:, :] = np.ma.masked_invalid(array_cyc_centroids)


    anticyclonic_shape = ds.createVariable('anticyc_shape', 'i1', ('lat', 'lon',), fill_value=fill_value)
    anticyclonic_shape.units = 'W'
    anticyclonic_shape.description = 'Array with the shapes of the anticyclonic eddies for that day'
    anticyclonic_shape[:, :] = np.ma.masked_invalid(array_anti_shape)

    anticyclonic_centroids = ds.createVariable('anticyc_centr', 'i1', ('lat', 'lon',), fill_value=fill_value)
    anticyclonic_centroids.units = 'Z'
    anticyclonic_centroids.description = 'Array with the dots with the centroids of the identified anticyclonic eddies for a week of movement'
    anticyclonic_centroids[:, :] = np.ma.masked_invalid(array_anti_centroids)

    lats[:] = np.linspace(33, 46, height)
    lons[:] = np.linspace(-20, -4, width)

    ds.close() 
    
    
############################################################# MAIN FUNCTION ################################################################


def main():
    data_directory = "/home/colabatlantic2/projects/JUNO/data/AVISO_data"
    output_directory = "/home/colabatlantic2/projects/JUNO/data/AVISO_netcdf"

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    eddies_user = os.environ.get("EDDIES_USER")
    eddies_pass = os.environ.get("EDDIES_PASS")
    if not eddies_user or not eddies_pass:
        raise RuntimeError("EDDIES_USER or EDDIES_PASS is not defined")

    cyclonic_name, anticyclonic_name = request_eddy_filenames(eddies_user, eddies_pass)

    latest_match = re.search(r"_(\d{8})\.nc$", cyclonic_name)
    if latest_match is None:
        raise RuntimeError(f"Could not extract the latest date from {cyclonic_name}")
    latest_aviso_date = datetime.datetime.strptime(latest_match.group(1), "%Y%m%d")

    existing_dates = set()
    for filename in os.listdir(output_directory):
        match = re.fullmatch(r"eddies_(\d{8})\.nc", filename)
        if match:
            existing_dates.add(datetime.datetime.strptime(match.group(1), "%Y%m%d"))

    # Preserve the historical start of the existing collection. If the output
    # directory is empty, create only the newest available day.
    first_date = min(existing_dates) if existing_dates else latest_aviso_date
    missing_dates = []
    candidate_date = first_date
    while candidate_date <= latest_aviso_date:
        if candidate_date not in existing_dates:
            missing_dates.append(candidate_date)
        candidate_date += timedelta(days=1)

    print(f"First local date: {first_date:%Y-%m-%d}", flush=True)
    print(f"Latest AVISO date: {latest_aviso_date:%Y-%m-%d}", flush=True)
    print(f"Missing dates to process: {len(missing_dates)}", flush=True)

    if not missing_dates:
        print("No missing dates. Nothing to process.", flush=True)
        return

    download_eddie(cyclonic_name, data_directory)
    download_eddie(anticyclonic_name, data_directory)

    cyclonic_slice_name = os.path.splitext(cyclonic_name)[0] + "_slice.nc"
    anticyclonic_slice_name = (
        os.path.splitext(anticyclonic_name)[0] + "_slice.nc"
    )

    # The complete downloaded files are no longer needed after ncks creates
    # the reduced files.
    for filename in (cyclonic_name, anticyclonic_name):
        full_path = os.path.join(data_directory, filename)
        if os.path.exists(full_path):
            os.remove(full_path)

    try:
        for position, data_final in enumerate(missing_dates, start=1):
            data_final_str = data_final.strftime("%Y%m%d")
            print(
                f"Processing {data_final_str} "
                f"({position}/{len(missing_dates)})",
                flush=True,
            )

            slice_netcdf(
                filepath=data_directory + "/",
                input_filename=cyclonic_slice_name,
                lat_min=35,
                lat_max=45,
                lon_min=330,
                lon_max=360,
                data_final_str=data_final_str,
                eddie_type="cyclonic",
            )
            slice_netcdf(
                filepath=data_directory + "/",
                input_filename=anticyclonic_slice_name,
                lat_min=35,
                lat_max=45,
                lon_min=330,
                lon_max=360,
                data_final_str=data_final_str,
                eddie_type="anticyclonic",
            )

            cyc = eddie_tracking(data_directory + "/", f"cyclonic_{data_final_str}.nc")
            anti = eddie_tracking(data_directory + "/", f"anticyclonic_{data_final_str}.nc")
            eddie_cyc_lats, eddie_cyc_lons, centro_cyc_y, centro_cyc_x = cyc
            eddie_anti_lats, eddie_anti_lons, centro_anti_y, centro_anti_x = anti

            arrays = eddies_arrays(
                eddie_cyc_lons,
                eddie_cyc_lats,
                centro_cyc_x,
                centro_cyc_y,
                eddie_anti_lons,
                eddie_anti_lats,
                centro_anti_x,
                centro_anti_y,
            )
            create_netcdf(data_final_str, *arrays)

            # These daily intermediate files can be recreated and consume
            # substantial space, so remove them after successful output.
            for prefix in ("cyclonic", "anticyclonic"):
                intermediate = os.path.join(
                    data_directory, f"{prefix}_{data_final_str}.nc"
                )
                if os.path.exists(intermediate):
                    os.remove(intermediate)
    finally:
        # Keep cleanup safe even when processing is interrupted or fails.
        for filename in (cyclonic_slice_name, anticyclonic_slice_name):
            sliced_path = os.path.join(data_directory, filename)
            if os.path.exists(sliced_path):
                os.remove(sliced_path)
        


if __name__ == '__main__':
    main()
    
    

        
