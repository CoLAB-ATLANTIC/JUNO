import xarray as xr
import pandas as pd
import os

# Function to modify NetCDF by adding time dimension
def add_time_dimension(file):
    # Extract date from filename (assumes YYYYMMDD00.nc format)
    filename = os.path.basename(file)
    date_str = filename[:8]  # Extract YYYYMMDD
    date = pd.to_datetime(date_str, format="%Y%m%d")  # Convert to datetime

    # Compute time reference (days since 1970-01-01)
    time_value = (date - pd.Timestamp("1970-01-01")).days

    # Open dataset
    ds = xr.open_dataset(file)
    # Add time dimension
    ds = ds.expand_dims({"time": [time_value]})  # Create a new time dimension

    # Set attributes for CF-compliance
    ds["time"].attrs["units"] = "days since 1970-01-01"
    ds["time"].attrs["calendar"] = "gregorian"
    new_file = file.replace(".nc", "_modified.nc")
    # Overwrite the original file
    ds.to_netcdf(new_file)  # Overwrites the file
    print(f"Updated: {new_file}")

# List all NetCDF files in the directory
import glob
file_list = glob.glob("2024*00.nc")  # Adjust path as needed

# Process each file
for file in file_list:
    try:
        add_time_dimension(file)
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("All files processed successfully!")

import os
import glob

# List all modified NetCDF files
modified_files = glob.glob("*_modified.nc")

for mod_file in modified_files:
    # Get the original filename by removing "modified_" prefix
    original_file = mod_file.replace("_modified", "")

    # Remove the old file
    if os.path.exists(original_file):
        os.remove(original_file)
        print(f"Deleted: {original_file}")

    # Rename the modified file to the original filename
    os.rename(mod_file, original_file)
    print(f"Renamed: {mod_file} → {original_file}")

print("All files replaced successfully!")
