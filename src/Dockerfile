# Python base Image according to the version on my conda environment
FROM python:3.8

# Set a working directory inside the container (typically its called app)
WORKDIR /app

# Copy my necessary scripts to the container
COPY MUR_daily_fronts_netcdf_docker.py /app/
COPY BOA.py /app/
COPY CayulaCornillon_xarray.py /app/

# Install libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx


# Install Python dependencies taking into consideration the libraries required to run the code
RUN pip install numpy xarray matplotlib scipy netCDF4 requests opencv-python-headless wget pydap dask


# Command to run your script
CMD ["python", "MUR_daily_fronts_netcdf_docker.py"]
