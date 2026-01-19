import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def calculate_ke_latitude_duacs(data_dir):
    """
    Calculates KE latitudinal position for DUACS daily data in a single folder.
    """
    # Find all .nc files and sort them (filenames contain dates like 20150504)
    nc_files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    
    if not nc_files:
        print(f"No .nc files found in {data_dir}")
        return []

    print(f"Found {len(nc_files)} files. Starting processing...")
    results = []

    for i, file_path in enumerate(nc_files):
        try:
            ds = xr.open_dataset(file_path)
            
            # 1. Coordinate Setup & Domain Selection
            # Pre-select region (120E-180E, 20N-50N)
            ds_sub = ds.sel(longitude=slice(120, 180), latitude=slice(20, 50))
            
            if ds_sub.longitude.size == 0 or ds_sub.latitude.size == 0:
                continue

            # Meshgrids for masking logic
            lons_1d = ds_sub['longitude'].values
            lats_1d = ds_sub['latitude'].values
            Lons, Lats = np.meshgrid(lons_1d, lats_1d)
            
            # 2. Extract Variables
            # DUACS usually uses 'adt', 'ugos', 'vgos'
            var_adt = 'adt' if 'adt' in ds_sub else 'zos'
            var_u = 'ugos' if 'ugos' in ds_sub else 'uo'
            var_v = 'vgos' if 'vgos' in ds_sub else 'vo'
            
            # Extract Data (Time=0)
            adt = ds_sub[var_adt].isel(time=0).values 
            u = ds_sub[var_u].isel(time=0).values
            v = ds_sub[var_v].isel(time=0).values
            
            # Get Timestamp (safer to read from file than parse filename)
            t_val = ds_sub['time'].isel(time=0).values
            
            # Progress print every 50 files
            if i % 50 == 0:
                print(f"Processing {pd.to_datetime(t_val).date()}...")

            # 3. Calculate Velocity Magnitude
            vel_mag = np.sqrt(u**2 + v**2)

            # 4. Define KE Axis (Nakano Method + Eddy Filter)
            # Domain: 150E - 170E
            domain_mask = (Lons >= 150) & (Lons <= 170)
            
            adt_flat = adt[domain_mask]
            vel_flat = vel_mag[domain_mask]
            lat_flat = Lats[domain_mask]
            lon_flat = Lons[domain_mask] # Needed for Span Check
            
            # Filter NaNs
            valid_mask = np.isfinite(adt_flat) & np.isfinite(vel_flat)
            adt_flat = adt_flat[valid_mask]
            vel_flat = vel_flat[valid_mask]
            lat_flat = lat_flat[valid_mask]
            lon_flat = lon_flat[valid_mask]

            h_ke = np.nan
            if len(adt_flat) > 0:
                # Binning
                bin_step = 0.002
                min_adt = np.floor(adt_flat.min() / bin_step) * bin_step
                max_adt = np.ceil(adt_flat.max() / bin_step) * bin_step
                
                if min_adt < max_adt:
                    bins = np.arange(min_adt, max_adt + bin_step, bin_step)
                    bin_indices = np.digitize(adt_flat, bins)
                    
                    vel_sums = np.bincount(bin_indices, weights=vel_flat, minlength=len(bins)+1)
                    counts = np.bincount(bin_indices, minlength=len(bins)+1)
                    
                    with np.errstate(invalid='ignore'):
                        u_h = vel_sums / counts
                    
                    h_vals = bins 
                    u_h = u_h[1:len(h_vals)+1] 
                    
                    # Smoothing
                    window_size = int(0.2 / bin_step)
                    u_h_smooth = u_h.copy()
                    if window_size > 0 and len(u_h) > window_size:
                        valid_u = np.isfinite(u_h)
                        if np.sum(valid_u) > 0:
                            x_idxs = np.arange(len(u_h))
                            u_h_interp = np.interp(x_idxs, x_idxs[valid_u], u_h[valid_u])
                            u_h_smooth = uniform_filter1d(u_h_interp, size=window_size)

                    # Identify Max Velocity
                    if not (np.all(np.isnan(u_h_smooth)) or np.nanmax(u_h_smooth) == 0):
                        u_max = np.nanmax(u_h_smooth)
                        
                        peaks_mask = (np.diff(np.sign(np.diff(u_h_smooth))) < 0)
                        peaks_idx = peaks_mask.nonzero()[0] + 1
                        candidates = [idx for idx in peaks_idx if u_h_smooth[idx] > 0.9 * u_max]
                        if not candidates:
                            candidates = [np.nanargmax(u_h_smooth)]

                        # --- EDDY FILTERING: Longitudinal Span Check ---
                        valid_candidates = []
                        for idx in candidates:
                            h_candidate = h_vals[idx]
                            mask_h = (adt_flat >= h_candidate - bin_step/2) & (adt_flat < h_candidate + bin_step/2)
                            
                            if np.any(mask_h):
                                contour_lons = lon_flat[mask_h]
                                span = contour_lons.max() - contour_lons.min()
                                
                                # Criterion: Main jet must span at least 15 degrees
                                if span >= 15.0:
                                    valid_candidates.append(idx)
                        
                        if not valid_candidates:
                             for idx in candidates:
                                h_candidate = h_vals[idx]
                                mask_h = (adt_flat >= h_candidate - bin_step/2) & (adt_flat < h_candidate + bin_step/2)
                                if np.any(mask_h):
                                    if (lon_flat[mask_h].max() - lon_flat[mask_h].min()) >= 10.0:
                                        valid_candidates.append(idx)
                        
                        if not valid_candidates:
                            valid_candidates = candidates

                        # Select Southernmost Axis
                        best_idx = -1
                        min_mean_lat = 999.0
                        for idx in valid_candidates:
                            h_candidate = h_vals[idx]
                            mask_h = (adt_flat >= h_candidate - bin_step/2) & (adt_flat < h_candidate + bin_step/2)
                            if np.any(mask_h):
                                mean_lat = np.mean(lat_flat[mask_h])
                                if mean_lat < min_mean_lat:
                                    min_mean_lat = mean_lat
                                    best_idx = idx
                        
                        if best_idx != -1:
                            h_ke = h_vals[best_idx]

            # 5. Determine Latitude at 144E
            if np.isnan(h_ke):
                ke_lat = np.nan
            else:
                try:
                    # Interpolate strictly to 144.0 degrees
                    ds_144 = ds_sub.interp(longitude=144.0, method='linear')
                    
                    adt_144 = ds_144[var_adt].isel(time=0).values 
                    lat_144 = ds_144['latitude'].values
                    u_144 = ds_144[var_u].isel(time=0).values
                    v_144 = ds_144[var_v].isel(time=0).values
                    spd_144 = np.sqrt(u_144**2 + v_144**2)
                    
                    valid_144 = np.isfinite(adt_144)
                    adt_144 = adt_144[valid_144]
                    lat_144 = lat_144[valid_144]
                    spd_144 = spd_144[valid_144]
                    
                    if len(adt_144) < 2:
                        ke_lat = np.nan
                    else:
                        # Find crossings
                        crossings = []
                        for k in range(len(adt_144)-1):
                            v1, v2 = adt_144[k], adt_144[k+1]
                            if (h_ke - v1) * (h_ke - v2) <= 0:
                                frac = (h_ke - v1) / (v2 - v1 + 1e-9)
                                lat_c = lat_144[k] + frac * (lat_144[k+1] - lat_144[k])
                                spd_c = spd_144[k] + frac * (spd_144[k+1] - spd_144[k])
                                crossings.append((lat_c, spd_c))
                        
                        if len(crossings) == 1:
                            ke_lat = crossings[0][0]
                        elif len(crossings) > 1:
                            # Pick crossing with MAX SPEED
                            best_c = max(crossings, key=lambda x: x[1])
                            ke_lat = best_c[0]
                        else:
                            ke_lat = np.nan

                except Exception as e:
                    ke_lat = np.nan

            results.append({'time': t_val, 'ke_lat_144e': ke_lat, 'h_ke': h_ke})
            ds.close()
            
        except Exception as e:
            print(f"  Error processing {os.path.basename(file_path)}: {e}")
            continue
    
    return results

def plot_ke_timeseries_refined(csv_file):
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df_valid = df.dropna(subset=['ke_lat_144e'])
    if df_valid.empty: return

    # Statistics (Fallback to full if range missing)
    mask_93_11 = (df_valid['time'].dt.year >= 1993) & (df_valid['time'].dt.year <= 2011)
    df_ref = df_valid[mask_93_11]
    
    mean_ref = df_ref['ke_lat_144e'].mean() if not df_ref.empty else df_valid['ke_lat_144e'].mean()
    std_ref = df_ref['ke_lat_144e'].std() if not df_ref.empty else df_valid['ke_lat_144e'].std()

    start_shade = pd.Timestamp("2023-04-01")
    end_shade = pd.Timestamp("2024-08-31")
    
    y_min = df_valid['ke_lat_144e'].min() - std_ref
    y_max = df_valid['ke_lat_144e'].max() + std_ref
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(start_shade, end_shade, color='#cce5ff', alpha=1.0, zorder=0)
    ax.plot(df['time'], df['ke_lat_144e'], color='red', linewidth=1.5, label='KE Latitude', zorder=2)
    ax.axhline(mean_ref, color='black', linestyle='-', linewidth=1, label='Mean', zorder=1)
    ax.axhline(mean_ref + 3*std_ref, color='black', linestyle='--', linewidth=1, label=r'3$\sigma$', zorder=1)
    ax.axhline(mean_ref - 3*std_ref, color='black', linestyle='--', linewidth=1, zorder=1)
    
    mask_shade = (df_valid['time'] >= start_shade) & (df_valid['time'] <= end_shade)
    if np.any(mask_shade):
        mean_shade = df_valid[mask_shade]['ke_lat_144e'].mean()
        ax.hlines(y=mean_shade, xmin=start_shade, xmax=end_shade, colors='blue', linewidth=2, zorder=3)

    ax.set_ylabel('Latitude')
    ax.set_title('(d) KE latitudinal position [ 144E ]', loc='left')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%dN'))
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Adjust X limits if needed
    if ax.get_xlim()[1] < mdates.date2num(end_shade):
        ax.set_xlim(right=mdates.date2num(end_shade) + 365)

    plt.tight_layout()
    plt.savefig('KE_latitudinal_position_144E_DUACS_final.png', dpi=300)
    print("Plot saved as KE_latitudinal_position_144E_DUACS_final.png")

# --- Main Execution ---
data_dir = "./daily"  # Set this to your flat folder path if different
output_csv = 'ke_latitude_144e_timeseries_duacs.csv'

all_results = calculate_ke_latitude_duacs(data_dir)

if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Processing complete. Saved to '{output_csv}'.")
    plot_ke_timeseries_refined(output_csv)
else:
    print("No results calculated.")
