#!/usr/bin/env python3
"""
Compare SCHISM profiles against TEAMS CTD observations stored in NPZ.
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from pylib import *
import numpy as np

NaN = np.nan

MPI = None
COMM = None
RANK = 0
SIZE = 1

USE_MPI = "--mpi" in sys.argv or os.environ.get("ENABLE_MPI", "0") == "1"
if "--mpi" in sys.argv:
    sys.argv.remove("--mpi")

if USE_MPI:
    try:
        from mpi4py import MPI

        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()
        SIZE = COMM.Get_size()
    except (ImportError, Exception) as exc:
        print(f"[WARN] MPI requested but initialization failed: {exc}. Falling back to serial mode.")
        MPI = None
        COMM = None
        RANK = 0
        SIZE = 1


CONFIG = {
    "coordinates": {
        "canonical": "180",  # use "180" for [-180, 180], "360" for [0, 360)
    },
    "date_range": {
        "start": (2017, 4, 1),
        "end": (2017, 12, 31),
    },
    "compare": {
        "location_only": None,  # True: ignore observation time and use a fixed SCHISM time
        "location_only_time": None,  # datenum or None to use first SCHISM time
        "station_ids": None, # ["4", "5","6", "7","8"],  # list of station_id strings to include, or None for all
        "station_names": None,  # list of station_name strings to include, or None for all
        "match_month_day": True,  # True: ignore year, match month/day across model times
    },
    "region": {
        "shapefile": "SO_bnd.shp",  # set to None to skip shapefile filtering
        "use_shapefile": True,
        "subset_bbox": None, #(141.48767,141.56841,38.41906,38.43688),  # (lon_min, lon_max, lat_min, lat_max)
    },
    "teams": {
        "npz_path": "/scratch2/08924/kmpark/post-proc/npz/onagawa_d1_ctd.npz",
        "lon_name": "lon",
        "lat_name": "lat",
        "time_name": "time",
        "depth_name": "depth",
        "temp_name": "temp",
        "salt_name": "sal",
        "station_id_name": "station_id",
        "station_name_name": "station_name",
    },

    "schism": [
        {
        "enabled": True,
        "label": "RUN01e",
        "color": "g",
        "run_dir": "/scratch2/08924/kmpark/RUN01e",
        "variables": ["temp", "salt"],
        "refdate": datenum("2017-1-2"),
        "stack_range": (1, 220),
        "stack_step": 1 / 24,
        "lon_mode": "180",  # set to "360" if SCHISM output uses 0-360 longitudes
        },
        {
        "enabled": True,
        "label": "RUN01d",
        "color": "b",
        "run_dir": "/scratch2/08924/kmpark/RUN01d",
        "variables": ["temp", "salt"],
        "refdate": datenum("2017-1-2"),
        "stack_range": (1, 220),
        "stack_step": 1 / 24,
        "lon_mode": "180",  # set to "360" if SCHISM output uses 0-360 longitudes
        },
    ],

    "global_model": {
        "enabled": True,  # set True to compare against a global model
        "label": "CMEMS",
        "color": "k",
        "data_dir": "/scratch2/08924/kmpark/CMEMS/Japan",
        "file_suffix": ".nc",
        "variables": {
            "lon": "longitude",
            "lat": "latitude",
            "depth": "depth",
            "temp": "thetao",
            "salt": "so",
        },
        "fill_value": -3e4,
        "lon_mode": "auto",  # "auto", "180", or "360"
        "search_radius": 1,  # grid cells to search for nearest wet point if target is all-NaN
    },
    "output": {
        "dir": "./CompTEAMS_RUN01bd_d1",
        "task_name": "ctd",
        "experiment_id": None,
        "write_task_metrics": True,
        "write_scatter_plots": True,
        "save_profile_plots": True,
        "metrics_raw_name": "CTD_metrics_raw.csv",
        "metrics_station_name": "CTD_stats.csv",
        "metrics_model_name": "CTD_stats_by_model.csv",
        "manifest_name": "CTD_manifest.json",
        "scatter_alpha": 0.7,
        "scatter_size": 9,
        "scatter_cmap": "jet_r",
        "scatter_depth_max": None,
    },
    "plot": {
        "linewidth": 1.5,
        "font_size": 7,
        "obs_color": "r",
        "map_xlim": (141.4127, 141.6027),  # e.g. (141.2, 141.8)
        "map_ylim": (38.3298, 38.4992),  # e.g. (38.2, 38.6)
    },
}


SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
rc("font", size=SMALL_SIZE)
rc("axes", titlesize=SMALL_SIZE)
rc("xtick", labelsize=SMALL_SIZE)
rc("ytick", labelsize=SMALL_SIZE)
rc("legend", fontsize=SMALL_SIZE)
rc("axes", labelsize=MEDIUM_SIZE)
rc("figure", titlesize=BIGGER_SIZE)

CANONICAL_LON_MODE = CONFIG["coordinates"].get("canonical", "180")


def _deep_update(base, override):
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare SCHISM profiles against TEAMS CTD observations.",
    )
    p.add_argument("--config", help="Optional JSON config overrides.")
    p.add_argument("--output-dir", help="Override CONFIG['output']['dir'].")
    p.add_argument("--experiment-id", help="Experiment ID written to output metrics.")
    p.add_argument("--teams-npz", help="Override TEAMS NPZ path.")
    p.add_argument("--disable-profile-plots", action="store_true", help="Skip profile/map plot generation.")
    p.add_argument("--disable-scatter", action="store_true", help="Skip integrated scatter plots.")
    p.add_argument("--disable-metrics", action="store_true", help="Skip writing standardized task metrics.")
    p.add_argument("--start-date", help="Override date_range.start as YYYY-MM-DD.")
    p.add_argument("--end-date", help="Override date_range.end as YYYY-MM-DD.")
    return p.parse_args(argv)


def _parse_ymd(text):
    y, m, d = [int(x) for x in str(text).strip().split("-")]
    return [y, m, d]


def _apply_runtime_overrides(args):
    global CONFIG, CANONICAL_LON_MODE

    cfg = copy.deepcopy(CONFIG)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = _deep_update(cfg, user_cfg)

    if args.output_dir:
        cfg.setdefault("output", {})
        cfg["output"]["dir"] = args.output_dir
    if args.experiment_id:
        cfg.setdefault("output", {})
        cfg["output"]["experiment_id"] = args.experiment_id
    if args.teams_npz:
        cfg.setdefault("teams", {})
        cfg["teams"]["npz_path"] = args.teams_npz
    if args.disable_profile_plots:
        cfg.setdefault("output", {})
        cfg["output"]["save_profile_plots"] = False
    if args.disable_scatter:
        cfg.setdefault("output", {})
        cfg["output"]["write_scatter_plots"] = False
    if args.disable_metrics:
        cfg.setdefault("output", {})
        cfg["output"]["write_task_metrics"] = False
    if args.start_date:
        cfg.setdefault("date_range", {})
        cfg["date_range"]["start"] = _parse_ymd(args.start_date)
    if args.end_date:
        cfg.setdefault("date_range", {})
        cfg["date_range"]["end"] = _parse_ymd(args.end_date)

    CONFIG = cfg
    CANONICAL_LON_MODE = CONFIG["coordinates"].get("canonical", "180")


def normalize_longitudes(lon, mode):
    if mode is None:
        return lon
    mode = str(mode).strip().lower()
    arr = np.asarray(lon, dtype=float)
    if mode in {"180", "-180", "[-180,180]", "[-180, 180]"}:
        arr = (arr + 180.0) % 360.0 - 180.0
    elif mode in {"360", "[0,360]", "[0,360)", "0-360"}:
        arr = np.mod(arr, 360.0)
    else:
        return lon
    if np.isscalar(lon):
        return float(arr)
    return arr


def normalize_bbox(bbox, mode):
    if bbox is None:
        return None
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_min = normalize_longitudes(lon_min, mode)
    lon_max = normalize_longitudes(lon_max, mode)
    return lon_min, lon_max, lat_min, lat_max


def rank_print(*args, **kwargs):
    if "flush" not in kwargs:
        kwargs["flush"] = True
    print(f"[Rank {RANK}]", *args, **kwargs)


def parse_date_range(date_cfg):
    start = datenum(*date_cfg["start"])
    end = datenum(*date_cfg["end"])
    return start, end


def setup_region(region_cfg):
    shapefile_path = region_cfg.get("shapefile")
    use_shapefile = region_cfg.get("use_shapefile", bool(shapefile_path))
    px = py = None
    if use_shapefile and shapefile_path:
        bp = read_shapefile_data(shapefile_path)
        px, py = bp.xy.T
        px = normalize_longitudes(px, CANONICAL_LON_MODE)
    bbox = region_cfg.get("subset_bbox")
    bbox = normalize_bbox(bbox, CANONICAL_LON_MODE)
    return {"px": px, "py": py, "bbox": bbox}


def point_in_region(region, lon, lat):
    lon = normalize_longitudes(lon, CANONICAL_LON_MODE)
    if region["px"] is not None and region["py"] is not None:
        inside = inside_polygon(array([[lon, lat]]), region["px"], region["py"]).ravel()[0]
        if not bool(inside):
            return False
    bbox = region["bbox"]
    if bbox:
        lon_min, lon_max, lat_min, lat_max = bbox
        if lon_min <= lon_max:
            lon_in_box = lon_min <= lon <= lon_max
        else:
            lon_in_box = lon >= lon_min or lon <= lon_max
        if not (lon_in_box and lat_min <= lat <= lat_max):
            return False
    return True


def drop_nan_pairs(depth, values):
    valid = (~isnan(depth)) & (~isnan(values))
    if valid.sum() == 0:
        return array([]), array([])
    return depth[valid], values[valid]


def select_obs_within_model_range(obs_depth, obs_temp, obs_salt, model_depth):
    valid = (~isnan(obs_depth)) & (~isnan(obs_temp)) & (~isnan(obs_salt))
    obs_depth = obs_depth[valid]
    obs_temp = obs_temp[valid]
    obs_salt = obs_salt[valid]
    if len(obs_depth) == 0:
        return None

    model_depth = array(model_depth, dtype=float)
    model_depth = model_depth[~isnan(model_depth)]
    if len(model_depth) == 0:
        return None

    depth_min = nanmin(model_depth)
    depth_max = nanmax(model_depth)
    within = (obs_depth >= depth_min) & (obs_depth <= depth_max)
    obs_depth = obs_depth[within]
    obs_temp = obs_temp[within]
    obs_salt = obs_salt[within]
    if len(obs_depth) == 0:
        return None
    return obs_depth, obs_temp, obs_salt


def compute_model_stats(model_depth, model_values, obs_depth, obs_values):
    pairs = interpolate_model_to_obs(model_depth, model_values, obs_depth, obs_values)
    if pairs is None:
        return None
    return pairs["stat"]


def compute_basic_metrics(obs, mod):
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(mod)
    if valid.sum() < 2:
        return {
            "n": int(valid.sum()),
            "bias": np.nan,
            "rmse": np.nan,
            "corr": np.nan,
            "obs_std": np.nan,
            "mod_std": np.nan,
            "nrmse_std": np.nan,
            "wss": np.nan,
            "crmsd": np.nan,
        }
    obs = obs[valid]
    mod = mod[valid]
    diff = mod - obs
    n = int(len(obs))
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    obs_std = float(np.std(obs))
    mod_std = float(np.std(mod))
    corr = float(np.corrcoef(obs, mod)[0, 1]) if n > 1 else np.nan
    nrmse = float(rmse / obs_std) if obs_std > 0 else np.nan
    obs_mean = float(np.mean(obs))
    denom = np.sum((np.abs(mod - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    wss = float(1.0 - np.sum((mod - obs) ** 2) / denom) if denom > 0 else np.nan
    obs_anom = obs - obs_mean
    mod_anom = mod - float(np.mean(mod))
    crmsd = float(np.sqrt(np.mean((mod_anom - obs_anom) ** 2)))
    return {
        "n": n,
        "bias": bias,
        "rmse": rmse,
        "corr": corr,
        "obs_std": obs_std,
        "mod_std": mod_std,
        "nrmse_std": nrmse,
        "wss": wss,
        "crmsd": crmsd,
    }


def interpolate_model_to_obs(model_depth, model_values, obs_depth, obs_values):
    model_depth, model_values = drop_nan_pairs(model_depth, model_values)
    obs_depth = np.asarray(obs_depth, dtype=float)
    obs_values = np.asarray(obs_values, dtype=float)
    if len(model_depth) < 2 or len(obs_depth) < 2:
        return None
    interp_fn = interpolate.interp1d(model_depth, model_values, bounds_error=False, fill_value=NaN)
    model_interp = interp_fn(obs_depth)
    valid = np.isfinite(model_interp) & np.isfinite(obs_values) & np.isfinite(obs_depth)
    if valid.sum() < 2:
        return None
    obs_valid = obs_values[valid]
    mod_valid = model_interp[valid]
    dep_valid = obs_depth[valid]
    return {
        "obs": obs_valid,
        "mod": mod_valid,
        "depth": dep_valid,
        "stat": get_stat(mod_valid, obs_valid),
        "metrics": compute_basic_metrics(obs_valid, mod_valid),
    }


def prepare_global_model(cfg, start, end):
    if not cfg.get("enabled", False):
        return None
    files = array([f for f in os.listdir(cfg["data_dir"]) if f.endswith(cfg.get("file_suffix", ".nc"))])
    if len(files) == 0:
        return None
    mti = array([datenum(*array(f.replace(".", "_").split("_")[1:5]).astype("int")) for f in files])
    fpt = (mti >= (start - 1)) * (mti < (end + 1))
    files = files[fpt]
    mti = mti[fpt]
    if len(files) == 0:
        return None
    sind = argsort(mti)
    cfg = cfg.copy()
    cfg.update({
        "files": files[sind],
        "times": mti[sind],
        "month_day": _month_day_array(mti[sind]),
        "cache_index": None,
        "cache_ds": None,
        "cache_meta": None,
    })
    return cfg


def load_global_dataset(cfg, index):
    if cfg["cache_index"] != index:
        if cfg["cache_ds"] is not None:
            try:
                cfg["cache_ds"].close()
            except Exception:
                pass
        cfg["cache_ds"] = ReadNC(os.path.join(cfg["data_dir"], cfg["files"][index]), 1)
        cfg["cache_index"] = index
        cfg["cache_meta"] = None

    if cfg.get("cache_meta") is None:
        ds = cfg["cache_ds"]
        vars_ = cfg["variables"]
        lon_data = np.asarray(ds.variables[vars_["lon"]][:], dtype=float).squeeze()
        lat_data = np.asarray(ds.variables[vars_["lat"]][:], dtype=float).squeeze()
        depth_data = np.asarray(ds.variables[vars_["depth"]][:], dtype=float).squeeze()
        meta = {
            "lon_raw": lon_data,
            "lat_raw": lat_data,
            "depth": depth_data,
            "lon_norm_180": normalize_longitudes(lon_data, "180"),
            "lon_norm_360": normalize_longitudes(lon_data, "360"),
            "grid_type": "1d" if lon_data.ndim == 1 and lat_data.ndim == 1 else "2d",
        }
        if meta["grid_type"] == "2d":
            meta["lon_grid_180"], meta["lat_grid"] = np.broadcast_arrays(meta["lon_norm_180"], lat_data)
            meta["lon_grid_360"], _ = np.broadcast_arrays(meta["lon_norm_360"], lat_data)
        cfg["cache_meta"] = meta
    return cfg["cache_ds"], cfg["cache_meta"]


def extract_global_profile(cfg, stime, lon, lat):
    if cfg is None:
        return None
    times = cfg["times"]
    idx = abs(times - stime).argmin()
    ds, meta = load_global_dataset(cfg, idx)
    vars_ = cfg["variables"]
    lon_data = meta["lon_raw"]
    lat_data = meta["lat_raw"]
    depth_data = meta["depth"]

    if lon_data.ndim > 2 or lat_data.ndim > 2:
        return None

    lon_mode = str(cfg.get("lon_mode", "auto")).lower()
    lon_canonical = normalize_longitudes(lon, CANONICAL_LON_MODE)

    def resolve_lon_norm(mode):
        if mode == "360":
            return meta["lon_norm_360"], normalize_longitudes(lon_canonical, "360")
        if mode in {"180", "canonical"}:
            return meta["lon_norm_180"], normalize_longitudes(lon_canonical, "180")
        if np.any(lon_data > 180):
            return meta["lon_norm_180"], normalize_longitudes(lon_canonical, "180")
        return lon_data.astype(float), lon_canonical

    lon_norm, target_lon = resolve_lon_norm(lon_mode)
    if lon_data.ndim == 1 and lat_data.ndim == 1:
        lon_idx = int(np.abs(lon_norm - target_lon).argmin())
        lat_idx = int(np.abs(lat_data - lat).argmin())
        lon_pick = lon_data[lon_idx]
        lat_pick = lat_data[lat_idx]
    else:
        if lon_mode == "360":
            lon_grid = meta["lon_grid_360"]
        elif lon_mode in {"180", "canonical"} or np.any(lon_data > 180):
            lon_grid = meta["lon_grid_180"]
        else:
            lon_grid, _ = np.broadcast_arrays(lon_norm, lat_data)
        lat_grid = meta["lat_grid"]
        dist = (lat_grid - lat) ** 2 + (lon_grid - target_lon) ** 2
        llidx = unravel_index(dist.argmin(), dist.shape)
        lat_idx, lon_idx = int(llidx[0]), int(llidx[1])
        lon_pick = lon_data[lat_idx, lon_idx]
        lat_pick = lat_data[lat_idx, lon_idx]

    def extract_profile(var, lat_i=lat_idx, lon_i=lon_idx):
        dims = list(var.dimensions)
        slices = [slice(None)] * var.ndim

        def axis_for(candidates, default=None):
            for name in candidates:
                if name in dims:
                    return dims.index(name)
            return default

        time_axis = axis_for(["time", "TIME", "t", "Time"], None)
        lat_axis = axis_for(["lat", "latitude", "LAT", "Latitude", "y", "yj", "nav_lat"], None)
        lon_axis = axis_for(["lon", "longitude", "LON", "Longitude", "x", "xi", "nav_lon"], None)

        if time_axis is not None:
            time_idx = int(np.clip(idx, 0, var.shape[time_axis] - 1))
            slices[time_axis] = time_idx
        if lat_axis is not None:
            lat_idx_clipped = int(np.clip(lat_i, 0, var.shape[lat_axis] - 1))
            slices[lat_axis] = lat_idx_clipped
        if lon_axis is not None:
            lon_idx_clipped = int(np.clip(lon_i, 0, var.shape[lon_axis] - 1))
            slices[lon_axis] = lon_idx_clipped

        data = var[tuple(slices)]
        if ma.isMaskedArray(data):
            data = data.filled(np.nan)
        data = np.asarray(data, dtype=float)
        data = np.squeeze(data)
        if data.ndim == 0:
            data = data.reshape(1)
        return data

    temp_var = ds.variables[vars_["temp"]]
    salt_var = ds.variables[vars_["salt"]]
    temp_profile = extract_profile(temp_var)
    salt_profile = extract_profile(salt_var)

    fill_value = cfg.get("fill_value")
    if fill_value is not None:
        temp_profile[np.isclose(temp_profile, fill_value, atol=1e-6, rtol=0)] = NaN
        salt_profile[np.isclose(salt_profile, fill_value, atol=1e-6, rtol=0)] = NaN

    temp_profile[np.abs(temp_profile) > 1e30] = NaN
    salt_profile[np.abs(salt_profile) > 1e30] = NaN

    def has_finite(arr):
        return np.isfinite(arr).any()

    search_radius = int(cfg.get("search_radius", 0))
    if search_radius > 0 and not (has_finite(temp_profile) or has_finite(salt_profile)):
        best = None
        if lon_data.ndim == 1 and lat_data.ndim == 1:
            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    li = lat_idx + di
                    lj = lon_idx + dj
                    if li < 0 or lj < 0 or li >= lat_data.size or lj >= lon_data.size:
                        continue
                    tpi = extract_profile(temp_var, lat_i=li, lon_i=lj)
                    spi = extract_profile(salt_var, lat_i=li, lon_i=lj)
                    if not (has_finite(tpi) or has_finite(spi)):
                        continue
                    lon_i = lon_norm[lj]
                    lat_i = lat_data[li]
                    dist2 = (lat_i - lat) ** 2 + (lon_i - target_lon) ** 2
                    if best is None or dist2 < best[0]:
                        best = (dist2, li, lj, tpi, spi)
        else:
            nlat, nlon = lat_data.shape
            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    li = lat_idx + di
                    lj = lon_idx + dj
                    if li < 0 or lj < 0 or li >= nlat or lj >= nlon:
                        continue
                    tpi = extract_profile(temp_var, lat_i=li, lon_i=lj)
                    spi = extract_profile(salt_var, lat_i=li, lon_i=lj)
                    if not (has_finite(tpi) or has_finite(spi)):
                        continue
                    lon_i = lon_norm[li, lj]
                    lat_i = lat_data[li, lj]
                    dist2 = (lat_i - lat) ** 2 + (lon_i - target_lon) ** 2
                    if best is None or dist2 < best[0]:
                        best = (dist2, li, lj, tpi, spi)
        if best is not None:
            _, li, lj, tpi, spi = best
            temp_profile = tpi
            salt_profile = spi
            if lon_data.ndim == 1:
                lon_pick = lon_data[lj]
                lat_pick = lat_data[li]
            else:
                lon_pick = lon_data[li, lj]
                lat_pick = lat_data[li, lj]

    return {
        "name": cfg["label"],
        "color": cfg["color"],
        "depth": depth_data,
        "temp": temp_profile,
        "salt": salt_profile,
        "time": cfg["times"][idx],
        "grid_lon": normalize_longitudes(lon_pick, CANONICAL_LON_MODE),
        "grid_lat": float(lat_pick),
    }

def prepare_schism(cfg):
    if isinstance(cfg, (list, tuple)):
        out = []
        for item in cfg:
            prepared = prepare_schism(item)
            if prepared is not None:
                out.append(prepared)
        return out
    if not cfg.get("enabled", False):
        return None
    cfg = cfg.copy()
    stack_start, stack_end = cfg["stack_range"]
    stack_step = cfg.get("stack_step", 1)
    stacks = arange(stack_start, stack_end + stack_step, stack_step)
    cfg["stacks"] = stacks
    cfg["mti"] = stacks - 1 + cfg["refdate"]
    cfg["month_day"] = _month_day_array(cfg["mti"])
    return cfg


def extract_schism_profile(cfg, stime, lon, lat):
    if cfg is None:
        return None
    tidx = abs(cfg["mti"] - stime).argmin()
    istack = int(cfg["stacks"][tidx])
    cvars = ["zcor", *cfg["variables"]]
    lon_mode = str(cfg.get("lon_mode", "canonical")).lower()
    if lon_mode in {"360", "[0,360)", "0-360"}:
        lon_query = normalize_longitudes(lon, "360")
    elif lon_mode in {"180", "canonical"}:
        lon_query = normalize_longitudes(lon, "180")
    else:
        lon_query = lon
    C = read_schism_output(cfg["run_dir"], cvars, c_[lon_query, lat], istack, fmt=1)
    sch_time = C.time + cfg["refdate"]
    tidx_prof = abs(sch_time - stime).argmin()
    depth = -C.zcor[tidx_prof, :]
    temp = C.temp[tidx_prof, :]
    salt = C.salt[tidx_prof, :]
    return {
        "name": cfg["label"],
        "color": cfg["color"],
        "depth": depth,
        "temp": temp,
        "salt": salt,
        "time": sch_time[tidx_prof],
    }


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def format_time_label(label, time_val):
    dt = num2date(time_val)
    return f"{label}:{dt.strftime('%Y-%m-%d %H:%M:%S')}"


def annotate_stats(ax, models, stats_key, font_size, xm, ym):
    dy = ym[1] - ym[0]
    dx = xm[1] - xm[0]
    y_cursor = ym[0] + 0.1 * dy
    for model in models:
        stat = model["stats"].get(stats_key)
        if stat is None:
            continue
        ax.text(xm[0] + 0.1 * dx, y_cursor, f"mdl={model['name']}", fontsize=font_size)
        ax.text(xm[0] + 0.1 * dx, y_cursor + 0.05 * dy, f"R={stat.R:.3f}", fontsize=font_size)
        ax.text(xm[0] + 0.1 * dx, y_cursor + 0.10 * dy, f"MAE={stat.MAE:.3f}", fontsize=font_size)
        ax.text(xm[0] + 0.1 * dx, y_cursor + 0.15 * dy, f"ME={stat.ME:.3f}", fontsize=font_size)
        ax.text(xm[0] + 0.1 * dx, y_cursor + 0.20 * dy, f"RMSE={stat.RMSD:.3f}", fontsize=font_size)
        y_cursor += 0.35 * dy


def log_skip(reason, profile_id=None, stime=None, lon=None, lat=None):
    parts = ["SKIP"]
    if profile_id is not None:
        parts.append(f"id={profile_id}")
    if stime is not None:
        parts.append(f"time={num2date(stime).strftime('%Y-%m-%d %H:%M:%S')}")
    if lon is not None and lat is not None:
        lon_norm = normalize_longitudes(lon, CANONICAL_LON_MODE)
        parts.append(f"lon={lon_norm:.2f},lat={lat:.2f}")
    parts.append(f"reason={reason}")
    rank_print(", ".join(parts))


def plot_profiles(meta, region, obs_plot, models, plot_cfg, output_dir):
    linewidth = plot_cfg["linewidth"]
    font_size = plot_cfg["font_size"]
    obs_color = plot_cfg["obs_color"]

    lon = meta["lon"]
    lat = meta["lat"]
    gm_lon = meta.get("gm_lon")
    gm_lat = meta.get("gm_lat")
    alon = meta["lon_array"]
    alat = meta["lat_array"]
    stime = meta["obs_time"]

    obs_time_str = num2date(stime).strftime("%Y-%m-%d %H:%M:%S")
    model_time_labels = ", ".join([format_time_label(m["name"], m["time"]) for m in models])

    depth_temp_obs, temp_obs = drop_nan_pairs(obs_plot["depth"], obs_plot["temp"])
    depth_salt_obs, salt_obs = drop_nan_pairs(obs_plot["depth"], obs_plot["salt"])
    if len(depth_temp_obs) == 0 or len(depth_salt_obs) == 0:
        return

    figure(figsize=[7.2, 3.5])
    clf()
    subplot(1, 2, 1)
    plot(alon, alat, f"{obs_color}+", markersize=10, label="Obs")
    plot(lon, lat, "b*", markersize=10, label="SCHISM")
    if gm_lon is not None and gm_lat is not None:
        plot(gm_lon, gm_lat, "k^", markersize=8, label="Global")
    if region["px"] is not None and region["py"] is not None:
        plot(region["px"], region["py"], "k")
    if plot_cfg.get("map_xlim") is not None:
        xlim(plot_cfg["map_xlim"])
    if plot_cfg.get("map_ylim") is not None:
        ylim(plot_cfg["map_ylim"])
    xlabel("Longitude")
    ylabel("Latitude")
    legend()

    subplot(1, 2, 2)
    temp_depth_sets = []
    temp_value_sets = []
    for model in models:
        m_depth, m_temp = drop_nan_pairs(model["depth"], model["temp"])
        if len(m_depth) == 0:
            continue
        plot(m_temp, m_depth, model["color"], lw=linewidth, label=model["name"])
        plot(m_temp, m_depth, marker="o", linestyle="None", markersize=3, color=model["color"])
        temp_depth_sets.append(m_depth)
        temp_value_sets.append(m_temp)
    plot(
        temp_obs,
        depth_temp_obs,
        color=obs_color,
        lw=linewidth,
        marker="o",
        markersize=4,
        label="Obs",
        zorder=5,
    )
    temp_depth_sets.append(depth_temp_obs)
    temp_value_sets.append(temp_obs)
    title(
        f"lon:{lon:.2f}, lat:{lat:.2f}, obs:{obs_time_str}, {model_time_labels}",
        fontsize=font_size,
        fontweight="bold",
    )
    temp_values_concat = concatenate(temp_value_sets)
    temp_depth_concat = concatenate(temp_depth_sets)
    xm = [nanmin(temp_values_concat) - 1, nanmax(temp_values_concat) + 1]
    ym = [nanmin(temp_depth_concat), nanmax(temp_depth_concat)]
    setp(gca(), ylim=ym, xlim=xm)
    gca().invert_yaxis()
    annotate_stats(gca(), models, "temp", font_size, xm, ym)
    legend(loc="lower right")
    xlabel("Temperature ($^\\circ$C)")
    ylabel("Depth (m)")
    gca().xaxis.grid("on")
    gca().yaxis.grid("on")
    gcf().tight_layout()

    savefig(
        os.path.join(
            output_dir,
            f"{meta['profile_id']}_temp_{num2date(stime).strftime('%Y%m%d%H%M%S')}.png",
        ),
        dpi=400,
        bbox_inches="tight",
    )
    close()

    figure(figsize=[7.2, 3.5])
    clf()
    subplot(1, 2, 1)
    plot(alon, alat, f"{obs_color}+", markersize=10, label="Obs")
    plot(lon, lat, "b*", markersize=10, label="SCHISM")
    if gm_lon is not None and gm_lat is not None:
        plot(gm_lon, gm_lat, "k^", markersize=8, label="Global")
    if region["px"] is not None and region["py"] is not None:
        plot(region["px"], region["py"], "k")
    if plot_cfg.get("map_xlim") is not None:
        xlim(plot_cfg["map_xlim"])
    if plot_cfg.get("map_ylim") is not None:
        ylim(plot_cfg["map_ylim"])
    xlabel("Longitude")
    ylabel("Latitude")
    legend()

    subplot(1, 2, 2)
    salt_depth_sets = []
    salt_value_sets = []
    for model in models:
        m_depth, m_salt = drop_nan_pairs(model["depth"], model["salt"])
        if len(m_depth) == 0:
            continue
        plot(m_salt, m_depth, model["color"], lw=linewidth, label=model["name"])
        plot(m_salt, m_depth, marker="o", linestyle="None", markersize=3, color=model["color"])
        salt_depth_sets.append(m_depth)
        salt_value_sets.append(m_salt)
    plot(
        salt_obs,
        depth_salt_obs,
        color=obs_color,
        lw=linewidth,
        marker="o",
        markersize=4,
        label="Obs",
        zorder=5,
    )
    salt_depth_sets.append(depth_salt_obs)
    salt_value_sets.append(salt_obs)
    title(
        f"lon:{lon:.2f}, lat:{lat:.2f}, obs:{obs_time_str}, {model_time_labels}",
        fontsize=font_size,
        fontweight="bold",
    )
    salt_values_concat = concatenate(salt_value_sets)
    salt_depth_concat = concatenate(salt_depth_sets)
    xm = [nanmin(salt_values_concat) - 1, nanmax(salt_values_concat) + 1]
    ym = [nanmin(salt_depth_concat), nanmax(salt_depth_concat)]
    setp(gca(), ylim=ym, xlim=xm)
    gca().invert_yaxis()
    annotate_stats(gca(), models, "salt", font_size, xm, ym)
    legend(loc="lower right")
    xlabel("Salinity (PSU)")
    ylabel("Depth (m)")
    gca().xaxis.grid("on")
    gca().yaxis.grid("on")
    gcf().tight_layout()

    savefig(
        os.path.join(
            output_dir,
            f"{meta['profile_id']}_salt_{num2date(stime).strftime('%Y%m%d%H%M%S')}.png",
        ),
        dpi=400,
        bbox_inches="tight",
    )
    close()


def build_profile_groups(station_ids, times):
    station_ids = np.asarray(station_ids).astype("U").ravel()
    time_str = np.asarray(times).astype("datetime64[s]").astype("U").ravel()
    if station_ids.shape != time_str.shape:
        raise ValueError(f"station_ids/time mismatch: {station_ids.shape} vs {time_str.shape}")
    keys = np.char.add(station_ids, np.char.add("|", time_str))
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    return uniq_keys, inv, time_str



def _month_day_array(times):
    return np.array([(num2date(t).month, num2date(t).day) for t in times], dtype=int)


def _pick_time_same_month_day(stime, times, month_day):
    if times is None or len(times) == 0:
        return stime
    obs = num2date(stime)
    mask = (month_day[:, 0] == obs.month) & (month_day[:, 1] == obs.day)
    if not np.any(mask):
        return float(times[abs(times - stime).argmin()])
    cand = times[mask]
    return float(cand[abs(cand - stime).argmin()])


def _sanitize_name(text):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "model"


def _write_csv_rows(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_metric_rows(pair_rows):
    station_groups = {}
    model_groups = {}
    for row in pair_rows:
        s_key = (
            str(row.get("model", "")),
            str(row.get("var", "")),
            str(row.get("station_id", "")),
            str(row.get("station_name", "")),
        )
        m_key = (
            str(row.get("model", "")),
            str(row.get("var", "")),
        )
        station_groups.setdefault(s_key, {"obs": [], "mod": []})
        model_groups.setdefault(m_key, {"obs": [], "mod": []})
        station_groups[s_key]["obs"].append(float(row.get("obs", np.nan)))
        station_groups[s_key]["mod"].append(float(row.get("model_value", np.nan)))
        model_groups[m_key]["obs"].append(float(row.get("obs", np.nan)))
        model_groups[m_key]["mod"].append(float(row.get("model_value", np.nan)))

    station_rows = []
    for (model, var, station_id, station_name), vals in sorted(station_groups.items()):
        metrics = compute_basic_metrics(vals["obs"], vals["mod"])
        station_rows.append(
            {
                "model": model,
                "task": "ctd",
                "var": var,
                "station_id": station_id,
                "station_id_full": station_id,
                "station_name": station_name,
                "n": metrics["n"],
                "bias": metrics["bias"],
                "rmse": metrics["rmse"],
                "corr": metrics["corr"],
                "obs_std": metrics["obs_std"],
                "mod_std": metrics["mod_std"],
                "nrmse_std": metrics["nrmse_std"],
                "wss": metrics["wss"],
                "crmsd": metrics["crmsd"],
            }
        )

    model_rows = []
    for (model, var), vals in sorted(model_groups.items()):
        metrics = compute_basic_metrics(vals["obs"], vals["mod"])
        model_rows.append(
            {
                "model": model,
                "task": "ctd",
                "var": var,
                "n": metrics["n"],
                "bias": metrics["bias"],
                "rmse": metrics["rmse"],
                "corr": metrics["corr"],
                "obs_std": metrics["obs_std"],
                "mod_std": metrics["mod_std"],
                "nrmse_std": metrics["nrmse_std"],
                "wss": metrics["wss"],
                "crmsd": metrics["crmsd"],
            }
        )
    return station_rows, model_rows


def _write_depth_scatter(pair_rows, output_cfg):
    outdir = output_cfg["dir"]
    by_model = {}
    for row in pair_rows:
        by_model.setdefault(str(row.get("model", "")), []).append(row)

    written = []
    for model_name, rows in sorted(by_model.items()):
        model_depth = np.asarray([float(r.get("depth", np.nan)) for r in rows], dtype=float)
        model_depth = np.abs(model_depth[np.isfinite(model_depth)])
        if model_depth.size == 0:
            continue

        depth_max_cfg = output_cfg.get("scatter_depth_max")
        if depth_max_cfg is None:
            depth_max = float(np.nanmax(model_depth))
        else:
            depth_max = float(depth_max_cfg)
        if not np.isfinite(depth_max) or depth_max <= 0:
            depth_max = float(np.nanmax(model_depth))
        depth_max = max(depth_max, 1e-6)

        fig, axs = subplots(1, 2, figsize=(10.5, 4.5))
        vars_info = [
            ("temp", "Temperature", r"$^\circ$C"),
            ("salt", "Salinity", "PSU"),
        ]
        mappable = None
        for ax, (var, title_txt, unit_txt) in zip(axs, vars_info):
            sub = [r for r in rows if str(r.get("var", "")) == var]
            if len(sub) == 0:
                ax.text(0.5, 0.5, f"No {var} pairs", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            x = np.asarray([float(r.get("model_value", np.nan)) for r in sub], dtype=float)
            y = np.asarray([float(r.get("obs", np.nan)) for r in sub], dtype=float)
            c = np.abs(np.asarray([float(r.get("depth", np.nan)) for r in sub], dtype=float))
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
            if valid.sum() < 2:
                ax.text(0.5, 0.5, f"Not enough {var} pairs", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            x = x[valid]
            y = y[valid]
            c = c[valid]
            c = np.clip(c, 0.0, depth_max)
            mappable = ax.scatter(
                x,
                y,
                c=c,
                s=float(output_cfg.get("scatter_size", 9)),
                alpha=float(output_cfg.get("scatter_alpha", 0.7)),
                cmap=str(output_cfg.get("scatter_cmap", "jet_r")),
                vmin=0.0,
                vmax=depth_max,
                edgecolors="none",
            )
            vmin = float(np.nanmin(np.r_[x, y]))
            vmax = float(np.nanmax(np.r_[x, y]))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = 0.0, 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            pad = 0.03 * (vmax - vmin)
            lo = vmin - pad
            hi = vmax + pad
            ax.plot([lo, hi], [lo, hi], "k", lw=1.7)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"Model ({unit_txt})")
            ax.set_ylabel(f"Observation ({unit_txt})")
            ax.set_title(title_txt)
            ax.grid(True, alpha=0.3)

            metrics = compute_basic_metrics(y, x)
            ax.text(
                0.03,
                0.96,
                f"ME: {metrics['bias']:.2f} {unit_txt}\nRMSE: {metrics['rmse']:.2f} {unit_txt}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=axs, orientation="horizontal", fraction=0.08, pad=0.14)
            cbar.set_label("Water depth (m)")

        fig.suptitle(f"{model_name}: model vs observation", fontsize=11)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        fp = os.path.join(outdir, f"{_sanitize_name(model_name)}_scatter_depth.png")
        savefig(fp, dpi=350, bbox_inches="tight")
        close(fig)
        written.append(fp)
    return written


def _write_task_outputs(pair_rows, output_cfg, run_summary):
    if len(pair_rows) == 0:
        payload = {
            "raw_file": None,
            "station_file": None,
            "model_file": None,
            "scatter_files": [],
        }
        manifest_file = os.path.join(output_cfg["dir"], output_cfg.get("manifest_name", "CTD_manifest.json"))
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump({"summary": run_summary, **payload, "raw_rows": 0, "station_rows": 0, "model_rows": 0}, f, indent=2)
        return payload

    outdir = output_cfg["dir"]
    do_metrics = bool(output_cfg.get("write_task_metrics", True))
    raw_file = None
    station_file = None
    model_file = None
    station_rows = []
    model_rows = []
    if do_metrics:
        raw_file = os.path.join(outdir, output_cfg.get("metrics_raw_name", "CTD_metrics_raw.csv"))
        station_file = os.path.join(outdir, output_cfg.get("metrics_station_name", "CTD_stats.csv"))
        model_file = os.path.join(outdir, output_cfg.get("metrics_model_name", "CTD_stats_by_model.csv"))
        raw_fields = [
            "task",
            "experiment_id",
            "model",
            "station_id",
            "station_id_full",
            "station_name",
            "profile_id",
            "obs_time",
            "var",
            "depth",
            "obs",
            "model_value",
            "error",
        ]
        _write_csv_rows(raw_file, pair_rows, raw_fields)

        station_rows, model_rows = _aggregate_metric_rows(pair_rows)
        station_fields = [
            "model",
            "task",
            "var",
            "station_id",
            "station_id_full",
            "station_name",
            "n",
            "bias",
            "rmse",
            "corr",
            "obs_std",
            "mod_std",
            "nrmse_std",
            "wss",
            "crmsd",
        ]
        model_fields = [
            "model",
            "task",
            "var",
            "n",
            "bias",
            "rmse",
            "corr",
            "obs_std",
            "mod_std",
            "nrmse_std",
            "wss",
            "crmsd",
        ]
        _write_csv_rows(station_file, station_rows, station_fields)
        _write_csv_rows(model_file, model_rows, model_fields)

    scatter_files = []
    if bool(output_cfg.get("write_scatter_plots", True)):
        scatter_files = _write_depth_scatter(pair_rows, output_cfg)

    manifest_file = os.path.join(outdir, output_cfg.get("manifest_name", "CTD_manifest.json"))
    payload = {
        "summary": run_summary,
        "raw_file": raw_file,
        "station_file": station_file,
        "model_file": model_file,
        "scatter_files": scatter_files,
        "raw_rows": len(pair_rows),
        "station_rows": len(station_rows),
        "model_rows": len(model_rows),
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main():
    start_t, end_t = parse_date_range(CONFIG["date_range"])
    region = setup_region(CONFIG["region"])
    output_cfg = CONFIG.get("output", {})

    sch_cfg = prepare_schism(CONFIG["schism"])
    gm_cfg = prepare_global_model(CONFIG["global_model"], start_t, end_t)
    sch_list = sch_cfg if isinstance(sch_cfg, list) else ([sch_cfg] if sch_cfg is not None else [])
    active_models = [cfg for cfg in ([gm_cfg] + sch_list) if cfg is not None]
    if len(active_models) == 0:
        rank_print("No models selected for comparison. Enable at least one model in CONFIG.")
        return

    if RANK == 0:
        ensure_output_dir(output_cfg["dir"])
    if MPI:
        COMM.Barrier()

    teams_cfg = CONFIG["teams"]
    S = loadz(teams_cfg["npz_path"])
    lon = np.asarray(getattr(S, teams_cfg["lon_name"]))
    lat = np.asarray(getattr(S, teams_cfg["lat_name"]))
    depth = np.asarray(getattr(S, teams_cfg["depth_name"]))
    temp = np.asarray(getattr(S, teams_cfg["temp_name"]))
    salt = np.asarray(getattr(S, teams_cfg["salt_name"]))
    st_id = np.asarray(getattr(S, teams_cfg["station_id_name"]))
    st_name = None
    st_name_field = teams_cfg.get("station_name_name")
    if st_name_field and hasattr(S, st_name_field):
        st_name = np.asarray(getattr(S, st_name_field))

    time_raw = np.asarray(getattr(S, teams_cfg["time_name"]))
    time_str = time_raw.astype("datetime64[s]").astype(str)
    obs_time = np.array([datenum(t) for t in time_str], dtype=float)

    uniq_keys, inv, _ = build_profile_groups(st_id, time_raw)
    total_profiles = len(uniq_keys)
    if total_profiles == 0:
        rank_print("No TEAMS profiles found.")
        return

    local_indices = [i for i in range(total_profiles) if i % SIZE == RANK]
    rank_print(f"Assigned {len(local_indices)} of {total_profiles} profiles.")

    compare_cfg = CONFIG.get("compare", {})
    location_only = bool(compare_cfg.get("location_only", False))
    location_only_time = compare_cfg.get("location_only_time")
    station_ids = compare_cfg.get("station_ids")
    station_names = compare_cfg.get("station_names")
    match_month_day = bool(compare_cfg.get("match_month_day", False))
    if station_ids is not None:
        station_ids = set(str(s) for s in station_ids)
    if station_names is not None:
        station_names = set(str(s) for s in station_names)
    if location_only and location_only_time is None:
        if sch_list:
            location_only_time = float(sch_list[0]["mti"][0])
        elif gm_cfg is not None:
            location_only_time = float(gm_cfg["times"][0])
        else:
            raise RuntimeError("location_only_time is None but no model time is available.")

    local_pair_rows = []
    local_summary = {
        "rank": RANK,
        "assigned_profiles": len(local_indices),
        "processed_profiles": 0,
        "skipped_profiles": 0,
        "saved_profile_plots": 0,
    }
    task_name = str(output_cfg.get("task_name", "ctd"))
    experiment_id = output_cfg.get("experiment_id")

    for profile_idx in local_indices:
        mask = inv == profile_idx
        if not np.any(mask):
            continue

        profile_id = str(uniq_keys[profile_idx])
        stime = obs_time[mask][0]
        lon_pt = float(np.nanmean(lon[mask]))
        lat_pt = float(np.nanmean(lat[mask]))
        station_id = str(st_id[mask][0]).strip()
        station_name = str(st_name[mask][0]).strip() if st_name is not None else ""
        if station_ids is not None and station_id not in station_ids:
            local_summary["skipped_profiles"] += 1
            log_skip("station_id filtered", profile_id, stime, lon_pt, lat_pt)
            continue
        if station_names is not None:
            if st_name is None:
                local_summary["skipped_profiles"] += 1
                log_skip("station_name unavailable", profile_id, stime, lon_pt, lat_pt)
                continue
            if station_name not in station_names:
                local_summary["skipped_profiles"] += 1
                log_skip("station_name filtered", profile_id, stime, lon_pt, lat_pt)
                continue
        if (not location_only) and (stime < start_t or stime > end_t):
            local_summary["skipped_profiles"] += 1
            log_skip("outside target window", profile_id, stime, lon_pt, lat_pt)
            continue
        if not point_in_region(region, lon_pt, lat_pt):
            local_summary["skipped_profiles"] += 1
            log_skip("outside configured region", profile_id, stime, lon_pt, lat_pt)
            continue

        obs_depth = np.asarray(depth[mask], dtype=float)
        obs_temp = np.asarray(temp[mask], dtype=float)
        obs_salt = np.asarray(salt[mask], dtype=float)
        valid = np.isfinite(obs_depth) & np.isfinite(obs_temp) & np.isfinite(obs_salt)
        if valid.sum() < 4:
            local_summary["skipped_profiles"] += 1
            log_skip("insufficient depth samples", profile_id, stime, lon_pt, lat_pt)
            continue

        obs_depth = obs_depth[valid]
        obs_temp = obs_temp[valid]
        obs_salt = obs_salt[valid]
        order = np.argsort(obs_depth)
        obs_depth = obs_depth[order]
        obs_temp = obs_temp[order]
        obs_salt = obs_salt[order]

        models = []
        gm_lon = None
        gm_lat = None
        if gm_cfg is not None:
            if location_only:
                stime_model = location_only_time
            elif match_month_day:
                stime_model = _pick_time_same_month_day(stime, gm_cfg["times"], gm_cfg.get("month_day"))
            else:
                stime_model = stime
            gm_profile = extract_global_profile(gm_cfg, stime_model, lon_pt, lat_pt)
            if gm_profile is not None:
                gm_lon = gm_profile.get("grid_lon")
                gm_lat = gm_profile.get("grid_lat")
                obs_subset = select_obs_within_model_range(obs_depth, obs_temp, obs_salt, gm_profile["depth"])
                if obs_subset is not None:
                    odi, otpi, osi = obs_subset
                    temp_cmp = interpolate_model_to_obs(gm_profile["depth"], gm_profile["temp"], odi, otpi)
                    salt_cmp = interpolate_model_to_obs(gm_profile["depth"], gm_profile["salt"], odi, osi)
                    if temp_cmp is not None or salt_cmp is not None:
                        gm_profile["stats"] = {
                            "temp": temp_cmp["stat"] if temp_cmp is not None else None,
                            "salt": salt_cmp["stat"] if salt_cmp is not None else None,
                        }
                        gm_profile["compare"] = {"temp": temp_cmp, "salt": salt_cmp}
                        models.append(gm_profile)
                    else:
                        log_skip("global model interpolation failed", profile_id, stime, lon_pt, lat_pt)
                else:
                    log_skip("global model depth range mismatch", profile_id, stime, lon_pt, lat_pt)
            else:
                log_skip("global model profile missing", profile_id, stime, lon_pt, lat_pt)

        for sch_item in sch_list:
            if location_only:
                stime_model = location_only_time
            elif match_month_day:
                stime_model = _pick_time_same_month_day(stime, sch_item["mti"], sch_item.get("month_day"))
            else:
                stime_model = stime
            sch_profile = extract_schism_profile(sch_item, stime_model, lon_pt, lat_pt)
            if sch_profile is not None:
                obs_subset = select_obs_within_model_range(obs_depth, obs_temp, obs_salt, sch_profile["depth"])
                if obs_subset is not None:
                    odi, otpi, osi = obs_subset
                    temp_cmp = interpolate_model_to_obs(sch_profile["depth"], sch_profile["temp"], odi, otpi)
                    salt_cmp = interpolate_model_to_obs(sch_profile["depth"], sch_profile["salt"], odi, osi)
                    if temp_cmp is not None or salt_cmp is not None:
                        sch_profile["stats"] = {
                            "temp": temp_cmp["stat"] if temp_cmp is not None else None,
                            "salt": salt_cmp["stat"] if salt_cmp is not None else None,
                        }
                        sch_profile["compare"] = {"temp": temp_cmp, "salt": salt_cmp}
                        models.append(sch_profile)
                    else:
                        log_skip("SCHISM interpolation failed", profile_id, stime, lon_pt, lat_pt)
                else:
                    log_skip("SCHISM depth range mismatch", profile_id, stime, lon_pt, lat_pt)
            else:
                log_skip("SCHISM profile missing", profile_id, stime, lon_pt, lat_pt)

        if len(models) == 0:
            local_summary["skipped_profiles"] += 1
            log_skip("no models available for comparison", profile_id, stime, lon_pt, lat_pt)
            continue

        local_summary["processed_profiles"] += 1
        obs_time_txt = num2date(stime).strftime("%Y-%m-%d %H:%M:%S")
        for model in models:
            model.setdefault("stats", {"temp": None, "salt": None})
            model_name = str(model.get("name", "model"))
            compares = model.get("compare", {})
            for var in ("temp", "salt"):
                cmp = compares.get(var)
                if cmp is None:
                    continue
                obs_vals = np.asarray(cmp["obs"], dtype=float)
                mod_vals = np.asarray(cmp["mod"], dtype=float)
                dep_vals = np.asarray(cmp["depth"], dtype=float)
                for j in range(len(obs_vals)):
                    local_pair_rows.append(
                        {
                            "task": task_name,
                            "experiment_id": experiment_id if experiment_id is not None else model_name,
                            "model": model_name,
                            "station_id": station_id,
                            "station_id_full": station_id,
                            "station_name": station_name,
                            "profile_id": profile_id,
                            "obs_time": obs_time_txt,
                            "var": var,
                            "depth": float(dep_vals[j]),
                            "obs": float(obs_vals[j]),
                            "model_value": float(mod_vals[j]),
                            "error": float(mod_vals[j] - obs_vals[j]),
                        }
                    )

        if bool(output_cfg.get("save_profile_plots", True)):
            obs_plot = {"depth": obs_depth, "temp": obs_temp, "salt": obs_salt}
            meta = {
                "lon": lon_pt,
                "lat": lat_pt,
                "lon_array": lon[mask],
                "lat_array": lat[mask],
                "obs_time": stime,
                "profile_id": profile_id,
                "gm_lon": gm_lon,
                "gm_lat": gm_lat,
            }
            plot_profiles(meta, region, obs_plot, models, CONFIG["plot"], output_cfg["dir"])
            local_summary["saved_profile_plots"] += 1

    local_summary["pair_samples"] = len(local_pair_rows)
    if MPI:
        pair_chunks = COMM.gather(local_pair_rows, root=0)
        summary_chunks = COMM.gather(local_summary, root=0)
    else:
        pair_chunks = [local_pair_rows]
        summary_chunks = [local_summary]

    if RANK == 0:
        pair_rows = [r for chunk in pair_chunks for r in chunk]
        total_summary = {
            "mpi_size": SIZE,
            "total_profiles": total_profiles,
            "assigned_profiles": int(sum(s["assigned_profiles"] for s in summary_chunks)),
            "processed_profiles": int(sum(s["processed_profiles"] for s in summary_chunks)),
            "skipped_profiles": int(sum(s["skipped_profiles"] for s in summary_chunks)),
            "saved_profile_plots": int(sum(s["saved_profile_plots"] for s in summary_chunks)),
            "pair_samples": int(sum(s["pair_samples"] for s in summary_chunks)),
        }
        _write_task_outputs(pair_rows, output_cfg, total_summary)
        rank_print("done")
    if MPI:
        COMM.Barrier()


if __name__ == "__main__":
    cli_args = _parse_args()
    _apply_runtime_overrides(cli_args)
    main()
    sys.exit()
