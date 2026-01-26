#!/usr/bin/env python3
"""
Processes raw JODC ocean current NetCDF files into a single, analysis-ready
.npz file compatible with pylib-based validation scripts.
"""

import os
from glob import glob
import numpy as np
from pylib import *

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Directory containing the JODC .nc files
JODC_DATA_DIR = './vector-140E-145E35N-40N'

# Output file path for the processed data
OUTPUT_NPZ_FILE = 'jodc_current_all.npz'

# Conversion factor from knots to meters per second
KNOTS_TO_MS = 0.514444

# --------------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------------
def main():
    """
    Main function to find, process, and aggregate JODC current data.
    """
    # Find all NetCDF files in the specified directory
    nc_files = glob(os.path.join(JODC_DATA_DIR, '**', '*.nc'), recursive=True)

    if not nc_files:
        print(f"Error: No .nc files found in '{JODC_DATA_DIR}'")
        return

    print(f"Found {len(nc_files)} NetCDF files to process.")

    # Initialize lists to store aggregated data
    all_time = []
    all_lon = []
    all_lat = []
    all_depth = []
    all_station = []
    all_speed = []
    all_direction = []

    # Process each NetCDF file
    for i, file_path in enumerate(nc_files):
        print(f"Processing file {i+1}/{len(nc_files)}: {os.path.basename(file_path)}")
        try:
            # Use pylib's ReadNC to load the data
            nc = ReadNC(file_path)

            # Extract data
            time_minutes = nc.variables['time'][:] 
            lon = nc.variables['longitude'][:] 
            lat = nc.variables['latitude'][:] 
            u = nc.variables['u'][:, 0]  # Data is on a single depth layer
            v = nc.variables['v'][:, 0]
            depth = nc.variables['depth'][:] 
            ship_code = nc.getncattr('SHIP_CODE').strip()

            # Handle fill values
            u[u > 99990] = np.nan
            v[v > 99990] = np.nan

            # Convert time from minutes since 1800-01-01 to datenum
            # datenum('1800-01-01') is a large negative number, so calculate offset from a known date
            base_date = datenum(1800, 1, 1)
            time_days = time_minutes / (60 * 24)
            time_datenum = base_date + time_days

            # Convert velocity from knots to m/s
            u_ms = u * KNOTS_TO_MS
            v_ms = v * KNOTS_TO_MS

            # Calculate speed and direction
            speed = np.sqrt(u_ms**2 + v_ms**2)
            direction = np.rad2deg(np.arctan2(u_ms, v_ms)) % 360

            # Create a unique station ID
            # Some files have moving coordinates, so we use the mean lat/lon
            mean_lon = np.nanmean(lon)
            mean_lat = np.nanmean(lat)
            station_id = f"{ship_code}_{mean_lon:.2f}_{mean_lat:.2f}"
            
            num_obs = len(time_datenum)
            station_ids = np.repeat(np.array([station_id]), num_obs)

            # Append to lists
            all_time.extend(time_datenum)
            all_lon.extend(lon)
            all_lat.extend(lat)
            all_depth.extend(np.repeat(depth, num_obs))
            all_station.extend(station_ids)
            all_speed.extend(speed)
            all_direction.extend(direction)

        except Exception as e:
            print(f"  --> Failed to process {os.path.basename(file_path)}: {e}")
            continue

    if not all_time:
        print("No data was successfully processed. Exiting.")
        return

    # Convert lists to numpy arrays
    S = zdata()
    S.time = np.array(all_time)
    S.lon = np.array(all_lon)
    S.lat = np.array(all_lat)
    S.depth = np.array(all_depth)
    S.station = np.array(all_station)
    S.spd = np.array(all_speed)
    S.dir = np.array(all_direction)

    # Sort data by time
    sort_idx = np.argsort(S.time)
    for key in S.__dict__.keys():
        S.__dict__[key] = S.__dict__[key][sort_idx]

    # Save to .npz file
    print(f"\nSaving processed data to '{OUTPUT_NPZ_FILE}'...")
    savez(OUTPUT_NPZ_FILE, S)
    print("Done.")

if __name__ == "__main__":
    main()
