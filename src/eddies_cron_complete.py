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
7. Rasterize eddy shapes and centroids in memory and write daily NetCDF files.
8. Remove intermediate files after successful processing to reduce storage usage.

The processing is resumable so after an interruption the script processes only the dates that are still missing.
"""



#Import libraries

import argparse
import os
import numpy as np
import xarray as xr
from shapely.geometry import Polygon
import datetime
from datetime import timedelta
import requests
import subprocess
import json
import re
import time
import xml.etree.ElementTree as ET
import netCDF4 as nc


DATE_FORMAT = "%Y%m%d"
DEFAULT_RECENT_WINDOW_DAYS = int(os.environ.get("EDDIES_RECENT_WINDOW_DAYS", "15"))


def get_base_dir():
    return os.environ.get("JUNO_BASE_DIR", "/app")


def get_data_dir():
    return os.environ.get("JUNO_DATA_DIR", os.path.join(get_base_dir(), "data"))


def get_aviso_dirs():
    data_dir = get_data_dir()
    return {
        "data": os.path.join(data_dir, "AVISO_data"),
        "netcdf": os.path.join(data_dir, "AVISO_netcdf"),
    }




################################ Usar subprocess para usar o python como se estivesse na command line ###############################################
######################################### Download the eddies file exactly how i wanted it ##########################################################


DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_RETRIES = int(os.environ.get("EDDIES_DOWNLOAD_RETRIES", "5"))
DOWNLOAD_RETRY_DELAY_SECONDS = int(os.environ.get("EDDIES_DOWNLOAD_RETRY_DELAY_SECONDS", "60"))
DOWNLOAD_TIMEOUT = (30, 600)
RASTER_WIDTH = int(os.environ.get("EDDIES_RASTER_WIDTH", "1200"))
RASTER_HEIGHT = int(os.environ.get("EDDIES_RASTER_HEIGHT", "1000"))
RASTER_LON_MIN = float(os.environ.get("EDDIES_RASTER_LON_MIN", "-20"))
RASTER_LON_MAX = float(os.environ.get("EDDIES_RASTER_LON_MAX", "-4"))
RASTER_LAT_MIN = float(os.environ.get("EDDIES_RASTER_LAT_MIN", "33"))
RASTER_LAT_MAX = float(os.environ.get("EDDIES_RASTER_LAT_MAX", "46"))
RASTER_LINE_RADIUS = int(os.environ.get("EDDIES_RASTER_LINE_RADIUS", "1"))
RASTER_POINT_RADIUS = int(os.environ.get("EDDIES_RASTER_POINT_RADIUS", "1"))
DEFAULT_AVISO_DATASET_PATH = ("dataset-duacs-nrt-value-added-eddy-trajectory")
AVISO_HOST = "https://tds-odatis.aviso.altimetry.fr"
AVISO_DATASET_PATH = os.environ.get("AVISO_EDDIES_DATASET_PATH", DEFAULT_AVISO_DATASET_PATH).strip("/")
AVISO_FILESERVER_URL = f"{AVISO_HOST}/thredds/fileServer/{AVISO_DATASET_PATH}"
CATALOG_URL = f"{AVISO_HOST}/thredds/catalog/{AVISO_DATASET_PATH}/catalog.xml"
EDDIES_FILE_GROUPS = tuple(
    group.strip().lower()
    for group in os.environ.get(
        "EDDIES_FILE_GROUPS",
        "long,short",
    ).split(",")
    if group.strip()
)

if not EDDIES_FILE_GROUPS:
    raise RuntimeError("EDDIES_FILE_GROUPS must contain at least one file group")


def parse_content_range_total(content_range):
    if not content_range:
        return None

    match = re.fullmatch(r"bytes \d+-\d+/(\d+|\*)", content_range.strip())
    if match is None or match.group(1) == "*":
        return None

    return int(match.group(1))


def download_with_resume(download_url, output_file, temporary_file, auth):
    last_error = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        resume_from = (
            os.path.getsize(temporary_file)
            if os.path.exists(temporary_file)
            else 0
        )
        headers = {}
        mode = "wb"

        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            mode = "ab"
            print(
                f"Resuming download from byte {resume_from} "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES})",
                flush=True,
            )
        else:
            print(
                f"Starting download attempt {attempt}/{DOWNLOAD_RETRIES}",
                flush=True,
            )

        try:
            with requests.get(
                download_url,
                auth=auth,
                headers=headers,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT,
            ) as response:
                if resume_from > 0 and response.status_code == 200:
                    print(
                        "AVISO did not honor the resume request; "
                        "restarting the download.",
                        flush=True,
                    )
                    resume_from = 0
                    mode = "wb"

                response.raise_for_status()

                expected_size = parse_content_range_total(
                    response.headers.get("Content-Range")
                )
                if expected_size is None and response.headers.get("Content-Length"):
                    expected_size = resume_from + int(response.headers["Content-Length"])

                with open(temporary_file, mode) as file:
                    for chunk in response.iter_content(
                        chunk_size=DOWNLOAD_CHUNK_SIZE
                    ):
                        if chunk:
                            file.write(chunk)

            actual_size = os.path.getsize(temporary_file)
            if expected_size is not None and actual_size != expected_size:
                raise RuntimeError(
                    f"Downloaded {actual_size} bytes, expected {expected_size}"
                )

            os.replace(temporary_file, output_file)
            return

        except (requests.RequestException, OSError, RuntimeError) as error:
            last_error = error
            print(
                f"Download attempt {attempt}/{DOWNLOAD_RETRIES} failed: {error}",
                flush=True,
            )

            if attempt < DOWNLOAD_RETRIES:
                time.sleep(DOWNLOAD_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Failed after {DOWNLOAD_RETRIES} download attempts. "
        f"Partial file kept for resume: {temporary_file}. Last error: {last_error}"
    )


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

    download_url = f"{AVISO_FILESERVER_URL}/{filename}"

    output_file = os.path.join(path, filename)
    temporary_file = output_file + ".part"

    sliced_file = os.path.join(path, f"{os.path.splitext(filename)[0]}_slice.nc")

    # Download the original file if it does not already exist
    if not os.path.exists(output_file):
        print(f"Downloading: {download_url}")
        print(f"Destination: {output_file}")

        try:
            download_with_resume(
                download_url,
                auth=(eddies_user, eddies_pass),
                output_file=output_file,
                temporary_file=temporary_file,
            )
            print(f"Download completed: {output_file}")

        except RuntimeError as error:
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
    
    file_path = os.path.join(filepath, input_filename)         #'test_slice.nc'
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

    netcdf_save_folder = os.path.join(
        filepath,
        eddie_type + '_' + data_final_str + '.nc',
    )
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
    
    netcdf_path = os.path.join(filepath, filename)
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



def empty_front_array():
    return np.full((RASTER_HEIGHT, RASTER_WIDTH), np.nan, dtype="float32")


def coordinate_to_pixel(lon, lat):
    col = round(
        (lon - RASTER_LON_MIN)
        / (RASTER_LON_MAX - RASTER_LON_MIN)
        * (RASTER_WIDTH - 1)
    )
    row = round(
        (lat - RASTER_LAT_MIN)
        / (RASTER_LAT_MAX - RASTER_LAT_MIN)
        * (RASTER_HEIGHT - 1)
    )

    return int(row), int(col)


def mark_pixel(array, row, col, radius):
    row_min = max(0, row - radius)
    row_max = min(array.shape[0], row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(array.shape[1], col + radius + 1)

    if row_min < row_max and col_min < col_max:
        array[row_min:row_max, col_min:col_max] = 1


def draw_line(array, start_lon, start_lat, end_lon, end_lat):
    start_row, start_col = coordinate_to_pixel(start_lon, start_lat)
    end_row, end_col = coordinate_to_pixel(end_lon, end_lat)

    steps = max(abs(end_row - start_row), abs(end_col - start_col)) + 1
    if steps <= 0:
        return

    rows = np.linspace(start_row, end_row, steps)
    cols = np.linspace(start_col, end_col, steps)

    for row, col in zip(rows, cols):
        mark_pixel(array, int(round(row)), int(round(col)), RASTER_LINE_RADIUS)


def rasterize_contours(contour_lons, contour_lats):
    array = empty_front_array()

    for lons, lats in zip(contour_lons, contour_lats):
        points = [
            (lon, lat)
            for lon, lat in zip(lons, lats)
            if np.isfinite(lon) and np.isfinite(lat)
        ]

        if len(points) == 1:
            row, col = coordinate_to_pixel(points[0][0], points[0][1])
            mark_pixel(array, row, col, RASTER_LINE_RADIUS)
            continue

        for (start_lon, start_lat), (end_lon, end_lat) in zip(points, points[1:]):
            draw_line(array, start_lon, start_lat, end_lon, end_lat)

    return array


def rasterize_points(point_lons, point_lats):
    array = empty_front_array()

    for lon, lat in zip(point_lons, point_lats):
        if not np.isfinite(lon) or not np.isfinite(lat):
            continue

        row, col = coordinate_to_pixel(lon, lat)
        mark_pixel(array, row, col, RASTER_POINT_RADIUS)

    return array


def eddies_arrays(eddie_cyc_lons, eddie_cyc_lats, centro_cyc_x, centro_cyc_y, eddie_anti_lons, eddie_anti_lats, centro_anti_x, centro_anti_y, file_last_date_str):
    
    """
    Rasterize eddy shapes and centroids into arrays without writing plots.
    """
    array_cyc_shape = rasterize_contours(eddie_cyc_lons, eddie_cyc_lats)
    array_cyc_centroids = rasterize_points(centro_cyc_x, centro_cyc_y)
    array_anti_shape = rasterize_contours(eddie_anti_lons, eddie_anti_lats)
    array_anti_centroids = rasterize_points(centro_anti_x, centro_anti_y)
    
    return array_cyc_shape, array_cyc_centroids, array_anti_shape, array_anti_centroids



def request_eddy_filenames(eddies_user, eddies_pass):
    """Return the configured cyclonic and anticyclonic AVISO filenames."""
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

    filenames = sorted(dict.fromkeys(filenames))

    def select_files(polarity):
        polarity = polarity.lower()
        selected = []
        polarity_matches = [
            name
            for name in filenames
            if polarity in name.lower()
            and (polarity != "cyclonic" or "anticyclonic" not in name.lower())
        ]

        for group in EDDIES_FILE_GROUPS:
            matches = [
                name
                for name in polarity_matches
                if group in name.lower()
            ]

            if matches:
                selected.append(max(matches))

        if not selected and polarity_matches:
            selected.append(max(polarity_matches))

        return selected

    cyclonic_files = select_files("cyclonic")
    anticyclonic_files = select_files("anticyclonic")

    if not cyclonic_files or not anticyclonic_files:
        raise RuntimeError("Could not identify both eddy files in the AVISO catalogue. " f"NetCDF files found: {filenames}")

    print(f"AVISO catalogue: {CATALOG_URL}", flush=True)
    print(f"Cyclonic files found: {cyclonic_files}", flush=True)
    print(f"Anticyclonic files found: {anticyclonic_files}", flush=True)
    return cyclonic_files, anticyclonic_files


def dates_from_filename(filename):
    dates = []

    for date_text in re.findall(r"\d{8}", filename):
        try:
            dates.append(datetime.datetime.strptime(date_text, "%Y%m%d"))
        except ValueError:
            pass

    return dates


def available_date_range(filenames):
    dates = []

    for filename in filenames:
        dates.extend(dates_from_filename(filename))

    if not dates:
        raise RuntimeError(f"Could not extract dates from AVISO filenames: {filenames}")

    return min(dates), max(dates)


def parse_yyyymmdd(date_text, label):
    try:
        return datetime.datetime.strptime(date_text, DATE_FORMAT)
    except ValueError as error:
        raise ValueError(f"{label} must use YYYYMMDD format, got {date_text!r}") from error


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate daily AVISO eddies NetCDF files. By default it processes "
            "only the latest AVISO date minus 15 days through the latest AVISO date."
        )
    )
    parser.add_argument(
        "--start-date",
        help="First day to process, in YYYYMMDD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Last day to process, in YYYYMMDD format.",
    )
    return parser.parse_args()


def configured_date(cli_value, env_name, label):
    if cli_value:
        return parse_yyyymmdd(cli_value, label)

    env_value = os.environ.get(env_name)
    if env_value:
        return parse_yyyymmdd(env_value, env_name)

    return None


def resolve_processing_range(args, aviso_start_date, latest_aviso_date):
    requested_start_date = configured_date(
        args.start_date,
        "EDDIES_START_DATE",
        "--start-date",
    )
    requested_end_date = configured_date(
        args.end_date,
        "EDDIES_END_DATE",
        "--end-date",
    )

    final_date = requested_end_date or latest_aviso_date

    if requested_start_date:
        first_date = requested_start_date
    else:
        first_date = final_date - timedelta(days=DEFAULT_RECENT_WINDOW_DAYS)
        if first_date < aviso_start_date:
            first_date = aviso_start_date

    if final_date > latest_aviso_date:
        raise RuntimeError(
            f"End date is later than the latest AVISO date "
            f"({latest_aviso_date:%Y%m%d})"
        )

    if first_date < aviso_start_date:
        raise RuntimeError(
            f"Start date is earlier than the first AVISO date "
            f"({aviso_start_date:%Y%m%d})"
        )

    if first_date > final_date:
        raise RuntimeError("Start date must be before or equal to end date")

    return first_date, final_date


def eddy_group_from_filename(filename):
    filename_lower = filename.lower()

    for group in EDDIES_FILE_GROUPS:
        if group in filename_lower:
            return group

    return os.path.splitext(os.path.basename(filename_lower))[0]


def process_eddy_sources(
    source_filenames,
    polarity,
    data_directory,
    data_final_str,
):
    all_eddy_lats = []
    all_eddy_lons = []
    all_centroid_y = []
    all_centroid_x = []

    for source_filename in source_filenames:
        source_slice_name = os.path.splitext(source_filename)[0] + "_slice.nc"
        source_group = eddy_group_from_filename(source_filename)
        eddie_type = f"{polarity}_{source_group}"

        slice_netcdf(
            filepath=data_directory,
            input_filename=source_slice_name,
            lat_min=35,
            lat_max=45,
            lon_min=330,
            lon_max=360,
            data_final_str=data_final_str,
            eddie_type=eddie_type,
        )

        eddy_lats, eddy_lons, centroid_y, centroid_x = eddie_tracking(
            data_directory,
            f"{eddie_type}_{data_final_str}.nc",
        )
        all_eddy_lats.extend(eddy_lats)
        all_eddy_lons.extend(eddy_lons)
        all_centroid_y.extend(centroid_y)
        all_centroid_x.extend(centroid_x)

        intermediate = os.path.join(
            data_directory,
            f"{eddie_type}_{data_final_str}.nc",
        )
        if os.path.exists(intermediate):
            os.remove(intermediate)

    return all_eddy_lats, all_eddy_lons, all_centroid_y, all_centroid_x



def create_netcdf(data_final_str, array_cyc_shape, array_cyc_centroids, array_anti_shape, array_anti_centroids):
    
    output_directory = get_aviso_dirs()["netcdf"]
    os.makedirs(output_directory, exist_ok=True)
    nc_file = os.path.join(output_directory, 'eddies_' + data_final_str + '.nc')

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
    args = parse_args()
    aviso_dirs = get_aviso_dirs()
    data_directory = aviso_dirs["data"]
    output_directory = aviso_dirs["netcdf"]

    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    eddies_user = os.environ.get("EDDIES_USER")
    eddies_pass = os.environ.get("EDDIES_PASS")
    if not eddies_user or not eddies_pass:
        raise RuntimeError("EDDIES_USER or EDDIES_PASS is not defined")

    cyclonic_names, anticyclonic_names = request_eddy_filenames(
        eddies_user,
        eddies_pass,
    )
    aviso_start_date, latest_aviso_date = available_date_range(
        cyclonic_names + anticyclonic_names
    )

    existing_dates = set()
    for filename in os.listdir(output_directory):
        match = re.fullmatch(r"eddies_(\d{8})\.nc", filename)
        if match:
            existing_dates.add(datetime.datetime.strptime(match.group(1), "%Y%m%d"))

    first_date, final_date = resolve_processing_range(
        args,
        aviso_start_date,
        latest_aviso_date,
    )

    missing_dates = []
    candidate_date = first_date
    while candidate_date <= final_date:
        if candidate_date not in existing_dates:
            missing_dates.append(candidate_date)
        candidate_date += timedelta(days=1)

    print(f"AVISO dataset path: {AVISO_DATASET_PATH}", flush=True)
    print(f"AVISO first available date: {aviso_start_date:%Y-%m-%d}", flush=True)
    print(f"Default recent window: {DEFAULT_RECENT_WINDOW_DAYS} days", flush=True)
    print(f"First date to process: {first_date:%Y-%m-%d}", flush=True)
    print(f"Last date to process: {final_date:%Y-%m-%d}", flush=True)
    print(f"Latest AVISO date: {latest_aviso_date:%Y-%m-%d}", flush=True)
    print(f"Missing dates to process: {len(missing_dates)}", flush=True)

    if not missing_dates:
        print("No missing dates. Nothing to process.", flush=True)
        return

    source_filenames = cyclonic_names + anticyclonic_names

    for source_filename in source_filenames:
        download_eddie(source_filename, data_directory)

    # The complete downloaded files are no longer needed after ncks creates
    # the reduced files.
    for filename in source_filenames:
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

            cyc = process_eddy_sources(
                cyclonic_names,
                "cyclonic",
                data_directory,
                data_final_str,
            )
            anti = process_eddy_sources(
                anticyclonic_names,
                "anticyclonic",
                data_directory,
                data_final_str,
            )
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
                file_last_date_str=data_final_str,
            )
            create_netcdf(data_final_str, *arrays)

    finally:
        # Keep cleanup safe even when processing is interrupted or fails.
        for filename in source_filenames:
            sliced_name = os.path.splitext(filename)[0] + "_slice.nc"
            sliced_path = os.path.join(data_directory, sliced_name)
            if os.path.exists(sliced_path):
                os.remove(sliced_path)

        for filename in os.listdir(data_directory):
            if not filename.endswith(".nc"):
                continue
            if not (
                filename.startswith("cyclonic_")
                or filename.startswith("anticyclonic_")
            ):
                continue
            sliced_path = os.path.join(data_directory, filename)
            if os.path.exists(sliced_path):
                os.remove(sliced_path)
        


if __name__ == '__main__':
    main()
    
    

        
