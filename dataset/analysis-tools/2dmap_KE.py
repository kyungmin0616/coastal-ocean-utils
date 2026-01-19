"""
2dmap_KC_unified.py - Unified Script for Generating Kuroshio Extension Axis Maps
(Refined with Temporal Tracking Algorithm)

CONFIGURATION OPTIONS EXPLAINED:
--------------------------------
1. dataset_name: Label for the dataset (MIROC, CMEMS, DUACS).
2. base_dir: Path to the root data directory.
3. file_structure: 'separate_files_yearly', 'single_file_yearly', 'single_file_flat'.
4. vars: Variable mapping ('adt', 'u', 'v').
5. dims: Dimension mapping ('lon', 'lat', 'time', 'depth').
6. unit_scale_factor: 0.01 for cm -> m, 1.0 for m -> m.
7. grid_type: '2d' (curvilinear) or '1d' (regular).
8. TIME_WINDOW: ('YYYY-MM-DD', 'YYYY-MM-DD') or None.
9. TRACKING_PENALTY: Cost per degree of latitude jump (prevents hopping).
"""

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob
import re
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

# --- ANALYSIS PARAMETERS ---
TIME_WINDOW = ('1993-01-01', '2022-12-31') 
# TIME_WINDOW = None 

# Tracking / Stability Parameters
TRACKING_PENALTY = 0.45  # m/s penalty per degree of jump
LAT_WINDOW = (30.0, 42.0) # Allowed latitude range at 144E

# Output Directory
OUTPUT_DIR = f"KE_Axis_Maps_Refined2_{CONFIG['dataset_name']}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def load_data_2d(item, t_idx, config):
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
                if v_type == 'adt': v_name = next((x for x in ['ssh', 'zos', 'adt'] if x in ds_obj), v_name)
                if v_type == 'u': v_name = next((x for x in ['uo', 'ugos'] if x in ds_obj), v_name)
                if v_type == 'v': v_name = next((x for x in ['vo', 'vgos'] if x in ds_obj), v_name)
            
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
        
        vel_mag = np.sqrt(u**2 + v**2)
        
        if vel_mag.shape == adt.shape:
            land_mask = np.isnan(adt)
            vel_mag[land_mask] = np.nan

        if item['type'] == 'multi':
            ds_adt.close(); ds_u.close(); ds_v.close()
        else:
            ds.close()

        return adt, vel_mag, Lons, Lats, t_val

    except Exception as e:
        print(f"  Error loading data: {e}")
        return None

def process_and_plot(config):
    queue = get_file_list(config)
    prev_lat = None # History tracking

    for item in queue:
        try:
            if item['type'] == 'multi':
                with xr.open_dataset(item['paths']['adt']) as tmp: n_times = tmp.sizes[config['dims']['time']]
            else:
                with xr.open_dataset(item['path']) as tmp: n_times = tmp.sizes[config['dims']['time']]
        except: continue

        for t_idx in range(n_times):
            data = load_data_2d(item, t_idx, config)
            if not data: continue
            
            adt, vel_mag, Lons, Lats, t_val = data
            timestamp = pd.to_datetime(t_val)
            date_str = timestamp.strftime("%Y-%m-%d")

            if TIME_WINDOW:
                start_dt = pd.to_datetime(TIME_WINDOW[0])
                end_dt = pd.to_datetime(TIME_WINDOW[1])
                if not (start_dt <= timestamp <= end_dt):
                    continue

            print(f"  Mapping {date_str}...")

            # --- ALGORITHM START: Nakano + Weighted Tracking ---
            domain_mask = (Lons >= 150) & (Lons <= 170)
            adt_flat = adt[domain_mask]
            vel_flat = vel_mag[domain_mask]
            
            valid = np.isfinite(adt_flat) & np.isfinite(vel_flat)
            adt_flat = adt_flat[valid]
            vel_flat = vel_flat[valid]
            lon_flat = Lons[domain_mask][valid]
            
            # 1. Identify Candidate ADT Levels (h_KE)
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
                    
                    window_size = int(0.2 / bin_step)
                    u_h_smooth = u_h[1:len(bins)+1]
                    
                    if window_size > 0 and len(u_h_smooth) > window_size:
                        mask_u = np.isfinite(u_h_smooth)
                        if mask_u.sum() > 0:
                            u_filled = np.interp(np.arange(len(u_h_smooth)), np.arange(len(u_h_smooth))[mask_u], u_h_smooth[mask_u])
                            u_h_smooth = uniform_filter1d(u_filled, size=window_size)

                    if not (np.all(np.isnan(u_h_smooth)) or np.nanmax(u_h_smooth) == 0):
                        u_max = np.nanmax(u_h_smooth)
                        peaks_mask = (np.diff(np.sign(np.diff(u_h_smooth))) < 0)
                        peaks_idx = peaks_mask.nonzero()[0] + 1
                        
                        # Gather valid peaks (Span > 10)
                        for idx in peaks_idx:
                            if u_h_smooth[idx] > 0.85 * u_max:
                                h_cand = bins[idx]
                                mask_h = (adt_flat >= h_cand - bin_step/2) & (adt_flat < h_cand + bin_step/2)
                                if np.any(mask_h):
                                    lons_c = lon_flat[mask_h]
                                    if (lons_c.max() - lons_c.min()) >= 10.0:
                                        candidates_h.append(h_cand)
                        
                        # Fallback
                        if not candidates_h:
                            candidates_h.append(bins[np.nanargmax(u_h_smooth)])

            # 2. Select Best h_KE using Tracking Score
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

                for h_cand in candidates_h:
                    for k in range(len(p_adts)-1):
                        v1, v2 = p_adts[k], p_adts[k+1]
                        if (h_cand - v1) * (h_cand - v2) <= 0:
                            frac = (h_cand - v1) / (v2 - v1 + 1e-9)
                            c_lat = p_lats[k] + frac * (p_lats[k+1] - p_lats[k])
                            c_spd = p_spds[k] + frac * (p_spds[k+1] - p_spds[k])
                            
                            if LAT_WINDOW[0] <= c_lat <= LAT_WINDOW[1]:
                                score = c_spd 
                                if prev_lat is not None:
                                    dist = abs(c_lat - prev_lat)
                                    score -= (dist * TRACKING_PENALTY)
                                
                                if score > best_score:
                                    best_score = score
                                    ke_lat = c_lat
                                    h_ke_final = h_cand

            # Update history
            if not np.isnan(ke_lat):
                prev_lat = ke_lat

            # --- ALGORITHM END ---

            # --- PLOTTING ---
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_facecolor('silver') 

            mesh = ax.pcolormesh(Lons, Lats, vel_mag, cmap='viridis', vmin=0, vmax=1.5, shading='auto')
            cbar = plt.colorbar(mesh, ax=ax, label='Speed (m/s)')
            
            if not np.isnan(h_ke_final):
                cs = ax.contour(Lons, Lats, adt, levels=[h_ke_final], colors='none')
                found_jet = False
                
                # Robust contour drawing (Geometric Filter on map)
                paths_list = []
                if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0:
                    for verts in cs.allsegs[0]: 
                        x_pts = verts[:, 0]
                        y_pts = verts[:, 1]
                        
                        span = x_pts.max() - x_pts.min()
                        min_lon = x_pts.min()
                        max_lon = x_pts.max()
                        
                        # Map Drawing Criteria:
                        # 1. Must span significant distance (>10 deg)
                        # 2. Must roughly cross the domain to be the main axis
                        #if span > 10.0 and min_lon < 155.0 and max_lon > 165.0:
                        #    ax.plot(x_pts, y_pts, color='magenta', linewidth=2.5)
                        #    found_jet = True
                        ax.plot(x_pts, y_pts, color='magenta', linewidth=2.5)
                        found_jet = True
                
                if found_jet:
                    # Legend
                    axis_patch = patches.Patch(color='magenta', label=f'KE Axis (ADT={h_ke_final:.2f}m)\nLat@144E={ke_lat:.2f}N')
                    ax.legend(handles=[axis_patch], loc='upper right')
                else:
                    ax.text(0.02, 0.95, "Axis fragmented", transform=ax.transAxes, color='white')

            ax.axvline(144, color='cyan', linestyle='-', linewidth=2)
            ax.text(144.2, ax.get_ylim()[0] + 1, "144E", color='cyan', fontweight='bold', va='bottom')

            ax.set_title(f'Kuroshio Extension Axis ({CONFIG["dataset_name"]}): {date_str}', fontsize=12)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal')
            
            filename = f"KE_Map_{timestamp.strftime('%Y%m%d')}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150)
            plt.close(fig)

if __name__ == "__main__":
    print(f"Starting 2D Mapping for: {CONFIG['dataset_name']}")
    print(f"Output Directory: {OUTPUT_DIR}")
    process_and_plot(CONFIG)
    print("All processing complete.")
