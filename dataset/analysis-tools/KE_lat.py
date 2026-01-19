"""
KE_lat_Unified.py - Unified Script with Temporal Tracking for Eddy Rejection

CONFIGURATION OPTIONS:
----------------------
1. dataset_name: Label (MIROC, CMEMS, DUACS).
2. base_dir: Path to data.
3. file_structure: 'separate_files_yearly', 'single_file_yearly', 'single_file_flat'.
4. TRACKING_PENALTY: The "cost" (in m/s) per degree of latitude jump. 
   - Higher values make the path "stickier" to its previous position.
   - Recommended: 0.15 (A 1-degree jump requires 0.15 m/s extra speed to justify).
"""

import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# --- USER CONFIGURATION SECTION ----------------------------------------------
# =============================================================================

# --- OPTION 1: MIROC ---
# CONFIG = {
#     'dataset_name': 'MIROC',
#     'base_dir': './out5r1', 
#     'file_structure': 'separate_files_yearly', 
#     'filename_pattern': None, 
#     'vars': {'adt': 'sho', 'u': 'uo', 'v': 'vo'},
#     'dims': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': 'lev'},
#     'unit_scale_factor': 0.01, 
#     'grid_type': '2d' 
# }

# --- OPTION 2: CMEMS GLORYS ---
CONFIG = {
    'dataset_name': 'CMEMS',
    'base_dir': './', 
    'file_structure': 'single_file_yearly', 
    'filename_pattern': '*.nc',
    'vars': {'adt': 'zos', 'u': 'uo', 'v': 'vo'},
    'dims': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': 'depth'},
    'unit_scale_factor': 1.0, 
    'grid_type': '1d'
}

# --- OPTION 3: DUACS/AVISO ---
#CONFIG = {
#     'dataset_name': 'DUACS',
#     'base_dir': './monthly', 
#     'file_structure': 'single_file_flat',
#     'filename_pattern': '*.nc',
#     'vars': {'adt': 'adt', 'u': 'ugos', 'v': 'vgos'},
#     'dims': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': None},
#     'unit_scale_factor': 1.0,
#     'grid_type': '1d'
#}

# --- TRACKING CONFIGURATION ---
# Penalty factor: How much velocity (m/s) is "worth" 1 degree of latitude jump?
# 0.15 means a jump of 4 degrees (eddy detach) requires the eddy to be 0.6 m/s faster than the jet to be picked.
TRACKING_PENALTY = 15 
LAT_WINDOW = (30.0, 42.0) # Hard limits for sanity

# =============================================================================
# --- CORE FUNCTIONS ----------------------------------------------------------
# =============================================================================

def get_file_list(config):
    file_queue = []
    base_dir = config['base_dir']

    # --- Helper: Extract Date for Sorting ---
    # Finds the first sequence of 6 or 8 digits (YYYYMM or YYYYMMDD)
    def extract_date(f):
        filename = os.path.basename(f)
        # Look for 8 digits (daily) or 6 digits (monthly)
        m = re.search(r'(\d{6,8})', filename)
        if m:
            return m.group(1)
        return "00000000" # Fallback to ensure it doesn't crash

    if config['file_structure'] == 'separate_files_yearly':
        # 1. MIROC Style: Sort Year Directories (y1980, y1981...)
        year_dirs = sorted(glob.glob(os.path.join(base_dir, "y*")))
        for yd in year_dirs:
            paths = {
                'adt': os.path.join(yd, 'shd_kc.nc'),
                'u': os.path.join(yd, 'uo_kc.nc'),
                'v': os.path.join(yd, 'vo_kc.nc')
            }
            if all(os.path.exists(p) for p in paths.values()):
                file_queue.append({'type': 'multi', 'paths': paths, 'context': yd})

    elif config['file_structure'] == 'single_file_yearly':
        # 2. CMEMS Style: Sort Year Folders, then Sort Files Chronologically
        # First, sort the year folders (e.g., ./1993, ./1994)
        year_dirs = sorted(glob.glob(os.path.join(base_dir, "[1-2][0-9][0-9][0-9]")))

        for yd in year_dirs:
            files = glob.glob(os.path.join(yd, config['filename_pattern']))
            # Explicitly sort by the date in the filename
            files = sorted(files, key=extract_date)

            for f in files:
                file_queue.append({'type': 'single', 'path': f, 'context': os.path.basename(f)})

    elif config['file_structure'] == 'single_file_flat':
        # 3. DUACS Style: Sort all files in one folder Chronologically
        all_files = glob.glob(os.path.join(base_dir, config['filename_pattern']))
        # Sort using the extract_date helper
        all_files = sorted(all_files, key=extract_date)

        for f in all_files:
            file_queue.append({'type': 'single', 'path': f, 'context': os.path.basename(f)})

    print(f"Found {len(file_queue)} items to process.")
    return file_queue

def load_data_slice(item, t_idx, config):
    try:
        if item['type'] == 'multi':
            ds_adt = xr.open_dataset(item['paths']['adt'])
            ds_u = xr.open_dataset(item['paths']['u'])
            ds_v = xr.open_dataset(item['paths']['v'])
            ds_grid = ds_adt 
        else:
            ds = xr.open_dataset(item['path'])
            ds_grid = ds.sel({config['dims']['lon']: slice(120, 180), config['dims']['lat']: slice(20, 50)})
            ds_adt = ds_grid
            ds_u = ds_grid
            ds_v = ds_grid

        t_val = ds_adt[config['dims']['time']].isel({config['dims']['time']: t_idx}).values

        def get_var(ds_obj, v_type):
            v_name = config['vars'][v_type]
            if v_name not in ds_obj:
                # Auto-fallback for common names
                if v_type == 'adt': v_name = next((x for x in ['ssh', 'zos', 'adt'] if x in ds_obj), v_name)
            
            data = ds_obj[v_name].isel({config['dims']['time']: t_idx})
            if config['dims']['depth'] and config['dims']['depth'] in data.dims:
                data = data.isel({config['dims']['depth']: 0})
            return data

        adt_da = get_var(ds_adt, 'adt')
        u_da = get_var(ds_u, 'u')
        v_da = get_var(ds_v, 'v')

        if config['dataset_name'] == 'MIROC':
            u_da = u_da.interp_like(adt_da, method='nearest')
            v_da = v_da.interp_like(adt_da, method='nearest')

        if config['grid_type'] == '1d':
            lons_1d = ds_grid[config['dims']['lon']].values
            lats_1d = ds_grid[config['dims']['lat']].values
            Lons, Lats = np.meshgrid(lons_1d, lats_1d)
        else:
            Lons = ds_grid[config['dims']['lon']].values
            Lats = ds_grid[config['dims']['lat']].values

        scale = config['unit_scale_factor']
        adt = adt_da.values * scale
        u = u_da.values * scale
        v = v_da.values * scale
        
        if item['type'] == 'multi':
            ds_adt.close(); ds_u.close(); ds_v.close()
        else:
            ds.close()

        return adt, u, v, Lons, Lats, t_val

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_ke_latitude(config):
    queue = get_file_list(config)
    results = []
    
    # *** TRACKING VARIABLE ***
    prev_lat = None 

    for i, item in enumerate(queue):
        try:
            if item['type'] == 'multi':
                with xr.open_dataset(item['paths']['adt']) as tmp: n_times = tmp.sizes[config['dims']['time']]
            else:
                with xr.open_dataset(item['path']) as tmp: n_times = tmp.sizes[config['dims']['time']]
        except: continue

        for t_idx in range(n_times):
            data = load_data_slice(item, t_idx, config)
            if not data: continue
            
            adt, u, v, Lons, Lats, t_val = data
            vel_mag = np.sqrt(u**2 + v**2)
            
            domain_mask = (Lons >= 150) & (Lons <= 170)
            adt_flat = adt[domain_mask]
            vel_flat = vel_mag[domain_mask]
            
            valid = np.isfinite(adt_flat) & np.isfinite(vel_flat)
            adt_flat = adt_flat[valid]
            vel_flat = vel_flat[valid]
            lon_flat = Lons[domain_mask][valid]
            lat_flat = Lats[domain_mask][valid]

            # --- 2. Find h_KE Candidates ---
            # We look for ALL strong velocity peaks, not just the global max
            candidates_h = []
            
            if len(adt_flat) > 0:
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
                    
                    # Smoothing
                    window_size = int(0.2 / bin_step)
                    u_h_smooth = u_h[1:len(bins)+1]
                    
                    if window_size > 0 and len(u_h_smooth) > window_size:
                        mask_u = np.isfinite(u_h_smooth)
                        if mask_u.sum() > 0:
                            u_filled = np.interp(np.arange(len(u_h_smooth)), np.arange(len(u_h_smooth))[mask_u], u_h_smooth[mask_u])
                            u_h_smooth = uniform_filter1d(u_filled, size=window_size)

                    # Find all local maxima > 0.85 * Umax (slightly looser to catch secondary jets)
                    if not (np.all(np.isnan(u_h_smooth)) or np.nanmax(u_h_smooth) == 0):
                        u_max = np.nanmax(u_h_smooth)
                        peaks_mask = (np.diff(np.sign(np.diff(u_h_smooth))) < 0)
                        peaks_idx = peaks_mask.nonzero()[0] + 1
                        
                        # Filter peaks
                        for idx in peaks_idx:
                            if u_h_smooth[idx] > 0.85 * u_max:
                                # Apply Span Check
                                h_cand = bins[idx]
                                mask_h = (adt_flat >= h_cand - bin_step/2) & (adt_flat < h_cand + bin_step/2)
                                if np.any(mask_h):
                                    lons_c = lon_flat[mask_h]
                                    # Relaxed span check (10 deg) to allow for meanders
                                    if (lons_c.max() - lons_c.min()) >= 10.0:
                                        candidates_h.append(h_cand)
                        
                        # If no valid candidates found, fallback to global max
                        if not candidates_h:
                            candidates_h.append(bins[np.nanargmax(u_h_smooth)])

            # --- 3. Determine Latitude at 144E (Weighted Selection) ---
            ke_lat = np.nan
            h_ke_final = np.nan
            
            best_score = -999.0
            
            # Prepare profile at 144E
            profile_lats = []
            profile_adts = []
            profile_spds = []
            
            for j in range(Lons.shape[0]):
                row_lons = Lons[j, :]
                if row_lons.min() < 144 < row_lons.max():
                    row_adt = adt[j, :]
                    row_lats = Lats[j, :]
                    row_spd = vel_mag[j, :]
                    
                    sort_idx = np.argsort(row_lons)
                    profile_adts.append(np.interp(144.0, row_lons[sort_idx], row_adt[sort_idx]))
                    profile_lats.append(np.interp(144.0, row_lons[sort_idx], row_lats[sort_idx]))
                    profile_spds.append(np.interp(144.0, row_lons[sort_idx], row_spd[sort_idx]))
            
            profile_adts = np.array(profile_adts)
            profile_lats = np.array(profile_lats)
            profile_spds = np.array(profile_spds)
            valid_p = np.isfinite(profile_adts)
            
            if len(profile_adts[valid_p]) > 1:
                p_adts = profile_adts[valid_p]
                p_lats = profile_lats[valid_p]
                p_spds = profile_spds[valid_p]

                # Evaluate ALL candidates
                for h_cand in candidates_h:
                    # Find all crossings for this h_cand
                    for k in range(len(p_adts)-1):
                        v1, v2 = p_adts[k], p_adts[k+1]
                        if (h_cand - v1) * (h_cand - v2) <= 0:
                            frac = (h_cand - v1) / (v2 - v1 + 1e-9)
                            c_lat = p_lats[k] + frac * (p_lats[k+1] - p_lats[k])
                            c_spd = p_spds[k] + frac * (p_spds[k+1] - p_spds[k])
                            
                            # Hard Window Check
                            if LAT_WINDOW[0] <= c_lat <= LAT_WINDOW[1]:
                                # --- SCORING LOGIC ---
                                score = c_spd # Base score is velocity
                                
                                # Apply Tracking Penalty if we have history
                                if prev_lat is not None:
                                    dist = abs(c_lat - prev_lat)
                                    penalty = dist * TRACKING_PENALTY
                                    score = score - penalty
                                
                                if score > best_score:
                                    best_score = score
                                    ke_lat = c_lat
                                    h_ke_final = h_cand

            # Update history ONLY if detection was successful
            if not np.isnan(ke_lat):
                prev_lat = ke_lat
            # Else: keep old prev_lat to bridge small gaps, or reset?
            # Keeping it bridges gaps (good for 1-month missing data).

            results.append({'time': t_val, 'ke_lat_144e': ke_lat, 'h_ke': h_ke_final})
            
            if i % 50 == 0 and t_idx == 0:
                ts_str = pd.to_datetime(t_val).strftime("%Y-%m-%d")
                print(f"Processed: {ts_str}, KE Lat: {ke_lat:.2f}")

    return results

def plot_results(csv_file):
    if not os.path.exists(csv_file): return
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'])
    df_valid = df.dropna(subset=['ke_lat_144e'])
    if df_valid.empty:
        print("No valid data to plot.")
        return

    mask_ref = (df_valid['time'].dt.year >= 1993) & (df_valid['time'].dt.year <= 2011)
    df_ref = df_valid[mask_ref]
    mean_ref = df_ref['ke_lat_144e'].mean() if not df_ref.empty else df_valid['ke_lat_144e'].mean()
    std_ref = df_ref['ke_lat_144e'].std() if not df_ref.empty else df_valid['ke_lat_144e'].std()

    start_shade = pd.Timestamp("2023-04-01")
    end_shade = pd.Timestamp("2024-08-31")
    mask_shade = (df_valid['time'] >= start_shade) & (df_valid['time'] <= end_shade)
    mean_shade = df_valid[mask_shade]['ke_lat_144e'].mean() if np.any(mask_shade) else np.nan

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(start_shade, end_shade, color='#cce5ff', alpha=1.0, zorder=0)
    ax.plot(df['time'], df['ke_lat_144e'], color='red', linewidth=1.5, zorder=2)
    
    ax.axhline(mean_ref, color='black', linestyle='-', linewidth=1, label='Mean', zorder=1)
    ax.axhline(mean_ref + 3*std_ref, color='black', linestyle='--', linewidth=1, label=r'3$\sigma$', zorder=1)
    ax.axhline(mean_ref - 3*std_ref, color='black', linestyle='--', linewidth=1, zorder=1)
    
    if not np.isnan(mean_shade):
        ax.hlines(y=mean_shade, xmin=start_shade, xmax=end_shade, colors='blue', linewidth=2, zorder=3)

    ax.set_title('(d) KE latitudinal position [ 144E ]', loc='left')
    ax.set_ylabel('Latitude')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%dN'))
    ax.set_ylim(df_valid['ke_lat_144e'].min() - std_ref, df_valid['ke_lat_144e'].max() + std_ref)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plot_name = 'KE_lat_Unified_Plot.png'
    plt.savefig(plot_name, dpi=300)
    print(f"Plot saved as {plot_name}")

if __name__ == "__main__":
    print(f"Starting analysis for: {CONFIG['dataset_name']}")
    data_results = calculate_ke_latitude(CONFIG)
    
    if data_results:
        out_csv = 'ke_lat_results.csv'
        df = pd.DataFrame(data_results)
        df.to_csv(out_csv, index=False)
        print(f"Saved results to {out_csv}")
        plot_results(out_csv)
    else:
        print("No results generated.")
