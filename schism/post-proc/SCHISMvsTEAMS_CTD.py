#!/usr/bin/env python3
"""
Compare SCHISM profiles against TEAMS CTD observations stored in NPZ.
"""

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CONFIG = {
    "model": {
        "coordinates": {
            "canonical": "180",  # use "180" for [-180, 180], "360" for [0, 360)
        },
        "source_mode": "npz",  # raw | npz
        "source_npz_paths": ["./npz/RUN01g_OB_D1.npz", "./npz/RUN03a_OB_D1.npz","./npz/RUN04a_OB_D1.npz", "./npz/RUN05a_OB_D1.npz"],
        "plot_depth_mode": "native",  # native | interp
        "schism": [
            {"enabled": True, "label": "RUN01g", "color": "k"},
            {"enabled": True, "label": "RUN03a", "color": "c"},
            {"enabled": True, "label": "RUN04a", "color": "b"},
            {"enabled": True, "label": "RUN05a", "color": "g"},
        ],
        "global": {
            "enabled": False,  # set True to compare against a global model
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
    },
    "obs": {
        "path": "./npz/onagawa_d1_ctd.npz",
        "fields": {
            "lon": "lon",
            "lat": "lat",
            "time": "time",
            "depth": "depth",
            "temp": "temp",
            "salt": "sal",
            "station_id": "station_id",
            "station_name": "station_name",
        },
    },
    "stations": {
        "ids": None,  # list of station_id strings to include, or None for all
        "names": None,  # list of station_name strings to include, or None for all
    },
    "time": {
        "start": (2012, 2, 1),
        "end": (2014, 12, 31),
        "location_only": None,  # True: ignore observation time and use a fixed SCHISM time
        "location_only_time": None,  # datenum or None to use first SCHISM time
        "match_month_day": None,  # True: ignore year, match month/day across model times
    },
    "map": {
        "region": {
            "shapefile": "./SO_bnd.shp",  # set to None to skip shapefile filtering
            "use_shapefile": True,
            "subset_bbox": None,  # (lon_min, lon_max, lat_min, lat_max)
        },
    },
    "output": {
        "dir": "./CompObs/CompTEAMS_OBD1_01g03a04a05a",
        "task_name": "ctd",
        "experiment_id": None,
        "write_task_metrics": True,
        "write_scatter_plots": True,
        "save_profile_plots": True,
        "plot_mode": "full",  # full | quick | off
        "quick_max_profiles_per_rank": 25,
        "quick_profile_stride": 5,
        "profile_dpi_full": 400,
        "profile_dpi_quick": 150,
        "profile_tight_layout_full": True,
        "profile_tight_layout_quick": False,
        "profile_bbox_inches_full": "tight",
        "profile_bbox_inches_quick": None,
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
        "map_auto_zoom": True,  # True: auto-center map on observation point
        "map_zoom_degree": 0.03,  # half-width in degrees around obs point (lon/lat)
        "map_zoom_degree_lon": None,  # optional lon half-width override
        "map_zoom_degree_lat": None,  # optional lat half-width override
        "map_xlim": (141.4127, 141.6027),  # e.g. (141.2, 141.8), it is not used when "map_auto_zoom": True
        "map_ylim": (38.3298, 38.4992),  # e.g. (38.2, 38.6), it is is not used when "map_auto_zoom": True
    },
}

# =============================================================================
# Imports
# =============================================================================
import argparse
import builtins
import copy as pycopy
import json
import os
import sys
import time
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy import interpolate
from pylib import (
    ReadNC,
    datenum,
    deep_update_dict,
    get_stat,
    init_mpi_runtime,
    inside_polygon,
    loadz,
    num2date,
    rank_log,
    read_schism_output,
    read_shapefile_data,
    report_work_assignment,
    compute_skill_metrics,
    write_csv_rows,
    read_csv_rows,
    write_rank_csv_chunk,
    collect_rank_csv_chunks,
    cleanup_rank_csv_chunks,
)
import numpy as np

NaN = np.nan


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

CONFIG = {}
CANONICAL_CONFIG = pycopy.deepcopy(DEFAULT_CONFIG)
CANONICAL_LON_MODE = DEFAULT_CONFIG.get("model", {}).get("coordinates", {}).get("canonical", "180")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# MPI runtime state
# =============================================================================
MPI, COMM, RANK, SIZE, USE_MPI = init_mpi_runtime(sys.argv)


def _deep_update(base, override):
    return deep_update_dict(base, override, merge_list_of_dicts=True)


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
    p.add_argument("--plot-mode", choices=["full", "quick", "off"], help="Profile plotting mode.")
    p.add_argument("--start-date", help="Override time.start as YYYY-MM-DD.")
    p.add_argument("--end-date", help="Override time.end as YYYY-MM-DD.")
    p.add_argument("--source-mode", choices=["raw", "npz"], help="Override model.source_mode.")
    p.add_argument("--source-npz", nargs="+", help="Override model.source_npz_paths when source-mode=npz.")
    p.add_argument("--plot-depth-mode", choices=["native", "interp"], help="Override model.plot_depth_mode.")
    return p.parse_args(argv)


def _parse_ymd(text):
    y, m, d = [int(x) for x in str(text).strip().split("-")]
    return [y, m, d]


def _canonical_to_runtime_config(canonical_cfg):
    model_cfg = pycopy.deepcopy(canonical_cfg.get("model", {}))
    obs_cfg = pycopy.deepcopy(canonical_cfg.get("obs", {}))
    fields_cfg = pycopy.deepcopy(obs_cfg.get("fields", {}))
    stations_cfg = pycopy.deepcopy(canonical_cfg.get("stations", {}))
    time_cfg = pycopy.deepcopy(canonical_cfg.get("time", {}))
    map_cfg = pycopy.deepcopy(canonical_cfg.get("map", {}))
    output_cfg = pycopy.deepcopy(canonical_cfg.get("output", {}))
    plot_cfg = pycopy.deepcopy(canonical_cfg.get("plot", {}))

    start_raw = time_cfg.get("start", [2012, 2, 1])
    end_raw = time_cfg.get("end", [2014, 12, 31])
    start_val = list(start_raw) if start_raw is not None else [2012, 2, 1]
    end_val = list(end_raw) if end_raw is not None else [2014, 12, 31]

    runtime_cfg = {
        "coordinates": pycopy.deepcopy(model_cfg.get("coordinates", {"canonical": "180"})),
        "date_range": {
            "start": start_val,
            "end": end_val,
        },
        "compare": {
            "station_ids": stations_cfg.get("ids"),
            "station_names": stations_cfg.get("names"),
            "location_only": time_cfg.get("location_only"),
            "location_only_time": time_cfg.get("location_only_time"),
            "match_month_day": time_cfg.get("match_month_day"),
        },
        "region": pycopy.deepcopy(map_cfg.get("region", {})),
        "source": {
            "mode": model_cfg.get("source_mode", "raw"),
            "npz_paths": pycopy.deepcopy(model_cfg.get("source_npz_paths")),
            "plot_depth_mode": model_cfg.get("plot_depth_mode", "native"),
        },
        "teams": {
            "npz_path": obs_cfg.get("path"),
            "lon_name": fields_cfg.get("lon", "lon"),
            "lat_name": fields_cfg.get("lat", "lat"),
            "time_name": fields_cfg.get("time", "time"),
            "depth_name": fields_cfg.get("depth", "depth"),
            "temp_name": fields_cfg.get("temp", "temp"),
            "salt_name": fields_cfg.get("salt", "sal"),
            "station_id_name": fields_cfg.get("station_id", "station_id"),
            "station_name_name": fields_cfg.get("station_name", "station_name"),
        },
        "schism": pycopy.deepcopy(model_cfg.get("schism", [])),
        "global_model": pycopy.deepcopy(model_cfg.get("global", {})),
        "output": output_cfg,
        "plot": plot_cfg,
    }
    return runtime_cfg


def _build_canonical_config(args):
    cfg = pycopy.deepcopy(DEFAULT_CONFIG)
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
        cfg.setdefault("obs", {})
        cfg["obs"]["path"] = args.teams_npz
    if args.disable_profile_plots:
        cfg.setdefault("output", {})
        cfg["output"]["save_profile_plots"] = False
    if args.disable_scatter:
        cfg.setdefault("output", {})
        cfg["output"]["write_scatter_plots"] = False
    if args.disable_metrics:
        cfg.setdefault("output", {})
        cfg["output"]["write_task_metrics"] = False
    if args.plot_mode:
        cfg.setdefault("output", {})
        cfg["output"]["plot_mode"] = str(args.plot_mode)
        if str(args.plot_mode).lower() == "off":
            cfg["output"]["save_profile_plots"] = False
    if args.start_date:
        cfg.setdefault("time", {})
        cfg["time"]["start"] = _parse_ymd(args.start_date)
    if args.end_date:
        cfg.setdefault("time", {})
        cfg["time"]["end"] = _parse_ymd(args.end_date)
    if args.source_mode:
        cfg.setdefault("model", {})
        cfg["model"]["source_mode"] = str(args.source_mode)
    if args.source_npz:
        cfg.setdefault("model", {})
        cfg["model"]["source_npz_paths"] = [str(x) for x in args.source_npz if str(x).strip()]
    if args.plot_depth_mode:
        cfg.setdefault("model", {})
        cfg["model"]["plot_depth_mode"] = str(args.plot_depth_mode)
    return cfg


def _apply_runtime_overrides(args):
    global CONFIG, CANONICAL_CONFIG, CANONICAL_LON_MODE

    canonical_cfg = _build_canonical_config(args)
    CONFIG = _canonical_to_runtime_config(canonical_cfg)
    CANONICAL_CONFIG = canonical_cfg
    CANONICAL_LON_MODE = CONFIG["coordinates"].get("canonical", "180")


if not CONFIG:
    CONFIG = _canonical_to_runtime_config(CANONICAL_CONFIG)
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
    rank0_only = bool(kwargs.pop("rank0_only", False))
    msg = " ".join(str(a) for a in args)
    rank_log(msg, rank=RANK, size=SIZE, rank0_only=rank0_only)


def _report_profile_assignment(tag, total_profiles, local_indices):
    report_work_assignment(
        tag,
        total_profiles,
        local_indices,
        rank=RANK,
        size=SIZE,
        comm=COMM,
        mpi_enabled=bool(MPI),
        logger=rank_print,
    )


def parse_date_range(date_cfg):
    start = datenum(*date_cfg["start"])
    end = datenum(*date_cfg["end"])
    return start, end


def setup_region(region_cfg):
    shapefile_path = region_cfg.get("shapefile")
    use_shapefile = region_cfg.get("use_shapefile", bool(shapefile_path))
    px = py = None
    if use_shapefile and shapefile_path:
        shp_candidates = [str(shapefile_path)]
        if not os.path.isabs(str(shapefile_path)):
            shp_candidates.append(os.path.join(SCRIPT_DIR, str(shapefile_path)))

        shp_path = None
        for cand in shp_candidates:
            if os.path.exists(cand):
                shp_path = cand
                break

        if shp_path is None:
            rank_print(
                f"[WARN] Shapefile not found: {shapefile_path}. "
                "Proceeding without shapefile region filter."
            )
        else:
            try:
                bp = read_shapefile_data(shp_path)
                px, py = bp.xy.T
                px = normalize_longitudes(px, CANONICAL_LON_MODE)
            except Exception as exc:
                rank_print(
                    f"[WARN] Failed to read shapefile '{shp_path}': {exc}. "
                    "Proceeding without shapefile region filter."
                )
    bbox = region_cfg.get("subset_bbox")
    bbox = normalize_bbox(bbox, CANONICAL_LON_MODE)
    return {"px": px, "py": py, "bbox": bbox}


def point_in_region(region, lon, lat):
    lon = normalize_longitudes(lon, CANONICAL_LON_MODE)
    if region["px"] is not None and region["py"] is not None:
        inside = inside_polygon(np.array([[lon, lat]]), region["px"], region["py"]).ravel()[0]
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
    valid = (~np.isnan(depth)) & (~np.isnan(values))
    if valid.sum() == 0:
        return np.array([]), np.array([])
    return depth[valid], values[valid]


def select_obs_within_model_range(obs_depth, obs_temp, obs_salt, model_depth):
    valid = (~np.isnan(obs_depth)) & (~np.isnan(obs_temp)) & (~np.isnan(obs_salt))
    obs_depth = obs_depth[valid]
    obs_temp = obs_temp[valid]
    obs_salt = obs_salt[valid]
    if len(obs_depth) == 0:
        return None

    model_depth = np.array(model_depth, dtype=float)
    model_depth = model_depth[~np.isnan(model_depth)]
    if len(model_depth) == 0:
        return None

    depth_min = np.nanmin(model_depth)
    depth_max = np.nanmax(model_depth)
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
    return compute_skill_metrics(obs, mod, min_n=2)


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
    files = np.array([f for f in os.listdir(cfg["data_dir"]) if f.endswith(cfg.get("file_suffix", ".nc"))])
    if len(files) == 0:
        return None
    mti = np.array([datenum(*np.array(f.replace(".", "_").split("_")[1:5]).astype("int")) for f in files])
    fpt = (mti >= (start - 1)) * (mti < (end + 1))
    files = files[fpt]
    mti = mti[fpt]
    if len(files) == 0:
        return None
    sind = np.argsort(mti)
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
        llidx = np.unravel_index(dist.argmin(), dist.shape)
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
        if np.ma.isMaskedArray(data):
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
    stacks = np.arange(stack_start, stack_end + stack_step, stack_step)
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
    C = read_schism_output(cfg["run_dir"], cvars, np.c_[lon_query, lat], istack, fmt=1)
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


def annotate_stats(ax, models, stats_key, font_size):
    lines = []
    for model in models:
        stat = model["stats"].get(stats_key)
        if stat is None:
            continue
        lines.append(f"{model['name']}  R={stat.R:.3f}  RMSE={stat.RMSD:.3f}")
    if not lines:
        return
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=font_size,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
    )


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


def _resolve_map_limits(meta, plot_cfg):
    if bool(plot_cfg.get("map_auto_zoom", False)):
        lon0 = _safe_float_scalar(meta.get("lon"), default=np.nan)
        lat0 = _safe_float_scalar(meta.get("lat"), default=np.nan)
        if np.isfinite(lon0) and np.isfinite(lat0):
            base = _safe_float_scalar(plot_cfg.get("map_zoom_degree", 0.03), default=0.03)
            zlon = _safe_float_scalar(plot_cfg.get("map_zoom_degree_lon"), default=base)
            zlat = _safe_float_scalar(plot_cfg.get("map_zoom_degree_lat"), default=base)
            zlon = max(float(zlon), 1e-6)
            zlat = max(float(zlat), 1e-6)
            return (lon0 - zlon, lon0 + zlon), (lat0 - zlat, lat0 + zlat)
    return plot_cfg.get("map_xlim"), plot_cfg.get("map_ylim")


def _prepare_plot_runtime(output_cfg):
    mode = str(output_cfg.get("plot_mode", "full")).strip().lower()
    if not bool(output_cfg.get("save_profile_plots", True)):
        mode = "off"
    rt = {
        "mode": mode,
        "enabled": mode != "off",
        "max_per_rank": int(output_cfg.get("quick_max_profiles_per_rank", 25)),
        "stride": max(1, int(output_cfg.get("quick_profile_stride", 5))),
    }
    if mode == "quick":
        rt["dpi"] = int(output_cfg.get("profile_dpi_quick", 150))
        rt["tight_layout"] = bool(output_cfg.get("profile_tight_layout_quick", False))
        rt["bbox_inches"] = output_cfg.get("profile_bbox_inches_quick", None)
    else:
        rt["dpi"] = int(output_cfg.get("profile_dpi_full", 400))
        rt["tight_layout"] = bool(output_cfg.get("profile_tight_layout_full", True))
        rt["bbox_inches"] = output_cfg.get("profile_bbox_inches_full", "tight")
    return rt


def _should_plot_profile(profile_index, n_saved, plot_runtime):
    if not bool(plot_runtime.get("enabled", False)):
        return False
    if str(plot_runtime.get("mode", "full")).lower() != "quick":
        return True
    if int(n_saved) >= int(plot_runtime.get("max_per_rank", 25)):
        return False
    stride = max(1, int(plot_runtime.get("stride", 5)))
    return (int(profile_index) % stride) == 0


def plot_profiles(meta, region, obs_plot, models, plot_cfg, output_dir, plot_runtime=None):
    linewidth = plot_cfg["linewidth"]
    font_size = plot_cfg["font_size"]
    obs_color = plot_cfg["obs_color"]
    if plot_runtime is None:
        plot_runtime = {"dpi": 400, "tight_layout": True, "bbox_inches": "tight"}

    lon = meta["lon"]
    lat = meta["lat"]
    gm_lon = meta.get("gm_lon")
    gm_lat = meta.get("gm_lat")
    alon = meta["lon_array"]
    alat = meta["lat_array"]
    stime = meta["obs_time"]

    obs_time_str = num2date(stime).strftime("%Y-%m-%d %H:%M:%S")
    if len(models) > 0:
        first_mdl_time = num2date(models[0]["time"]).strftime("%Y-%m-%d %H:%M:%S")
    else:
        first_mdl_time = "-"
    profile_title = f"lon:{lon:.2f}, lat:{lat:.2f}, obs:{obs_time_str}, mdl0:{first_mdl_time}"
    map_xlim, map_ylim = _resolve_map_limits(meta, plot_cfg)

    n_models = max(1, int(len(models)))
    fig_w = min(12.5, 7.2 + 0.55 * max(0, n_models - 2))

    depth_temp_obs, temp_obs = drop_nan_pairs(obs_plot["depth"], obs_plot["temp"])
    depth_salt_obs, salt_obs = drop_nan_pairs(obs_plot["depth"], obs_plot["salt"])
    if len(depth_temp_obs) == 0 or len(depth_salt_obs) == 0:
        return

    plt.figure(figsize=[fig_w, 3.5])
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(alon, alat, f"{obs_color}+", markersize=10, label="Obs")
    plt.plot(lon, lat, "b*", markersize=10, label="SCHISM")
    if gm_lon is not None and gm_lat is not None:
        plt.plot(gm_lon, gm_lat, "k^", markersize=8, label="Global")
    if region["px"] is not None and region["py"] is not None:
        plt.plot(region["px"], region["py"], "k")
    if map_xlim is not None:
        plt.xlim(map_xlim)
    if map_ylim is not None:
        plt.ylim(map_ylim)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    plt.subplot(1, 2, 2)
    temp_depth_sets = []
    temp_value_sets = []
    for model in models:
        m_depth, m_temp = drop_nan_pairs(model["depth"], model["temp"])
        if len(m_depth) == 0:
            continue
        plt.plot(m_temp, m_depth, model["color"], lw=linewidth, label=model["name"])
        plt.plot(m_temp, m_depth, marker="o", linestyle="None", markersize=3, color=model["color"])
        temp_depth_sets.append(m_depth)
        temp_value_sets.append(m_temp)
    plt.plot(
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
    plt.title(profile_title, fontsize=font_size, fontweight="bold")
    temp_values_concat = np.concatenate(temp_value_sets)
    temp_depth_concat = np.concatenate(temp_depth_sets)
    xm = [np.nanmin(temp_values_concat) - 1, np.nanmax(temp_values_concat) + 1]
    ym = [np.nanmin(temp_depth_concat), np.nanmax(temp_depth_concat)]
    plt.setp(plt.gca(), ylim=ym, xlim=xm)
    plt.gca().invert_yaxis()
    annotate_stats(plt.gca(), models, "temp", font_size)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.xlabel("Temperature ($^\\circ$C)")
    plt.ylabel("Depth (m)")
    plt.gca().xaxis.grid("on")
    plt.gca().yaxis.grid("on")
    if bool(plot_runtime.get("tight_layout", True)):
        plt.gcf().tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

    plt.savefig(
        os.path.join(
            output_dir,
            f"{meta['profile_id']}_temp_{num2date(stime).strftime('%Y%m%d%H%M%S')}.png",
        ),
        dpi=int(plot_runtime.get("dpi", 400)),
        bbox_inches=plot_runtime.get("bbox_inches", "tight"),
    )
    plt.close()

    plt.figure(figsize=[fig_w, 3.5])
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(alon, alat, f"{obs_color}+", markersize=10, label="Obs")
    plt.plot(lon, lat, "b*", markersize=10, label="SCHISM")
    if gm_lon is not None and gm_lat is not None:
        plt.plot(gm_lon, gm_lat, "k^", markersize=8, label="Global")
    if region["px"] is not None and region["py"] is not None:
        plt.plot(region["px"], region["py"], "k")
    if map_xlim is not None:
        plt.xlim(map_xlim)
    if map_ylim is not None:
        plt.ylim(map_ylim)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    plt.subplot(1, 2, 2)
    salt_depth_sets = []
    salt_value_sets = []
    for model in models:
        m_depth, m_salt = drop_nan_pairs(model["depth"], model["salt"])
        if len(m_depth) == 0:
            continue
        plt.plot(m_salt, m_depth, model["color"], lw=linewidth, label=model["name"])
        plt.plot(m_salt, m_depth, marker="o", linestyle="None", markersize=3, color=model["color"])
        salt_depth_sets.append(m_depth)
        salt_value_sets.append(m_salt)
    plt.plot(
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
    plt.title(profile_title, fontsize=font_size, fontweight="bold")
    salt_values_concat = np.concatenate(salt_value_sets)
    salt_depth_concat = np.concatenate(salt_depth_sets)
    xm = [np.nanmin(salt_values_concat) - 1, np.nanmax(salt_values_concat) + 1]
    ym = [np.nanmin(salt_depth_concat), np.nanmax(salt_depth_concat)]
    plt.setp(plt.gca(), ylim=ym, xlim=xm)
    plt.gca().invert_yaxis()
    annotate_stats(plt.gca(), models, "salt", font_size)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.xlabel("Salinity (PSU)")
    plt.ylabel("Depth (m)")
    plt.gca().xaxis.grid("on")
    plt.gca().yaxis.grid("on")
    if bool(plot_runtime.get("tight_layout", True)):
        plt.gcf().tight_layout(rect=[0.0, 0.0, 0.86, 1.0])

    plt.savefig(
        os.path.join(
            output_dir,
            f"{meta['profile_id']}_salt_{num2date(stime).strftime('%Y%m%d%H%M%S')}.png",
        ),
        dpi=int(plot_runtime.get("dpi", 400)),
        bbox_inches=plot_runtime.get("bbox_inches", "tight"),
    )
    plt.close()


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
    write_csv_rows(path, rows, fieldnames)


CTD_PAIR_FIELDS = [
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


def _read_csv_rows(path):
    return read_csv_rows(path)


def _write_pair_chunk(outdir, chunk_name, rank, rows):
    cdir, cpath = write_rank_csv_chunk(
        outdir,
        chunk_name,
        rank,
        rows,
        CTD_PAIR_FIELDS,
        prefix="pair",
    )
    return str(cdir), str(cpath)


def _collect_pair_chunks(cdir, nrank):
    return collect_rank_csv_chunks(cdir, nrank, prefix="pair")


def _cleanup_pair_chunks(cdir):
    cleanup_rank_csv_chunks(cdir, prefix="pair")


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

        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
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
            x = np.asarray([float(r.get("obs", np.nan)) for r in sub], dtype=float)
            y = np.asarray([float(r.get("model_value", np.nan)) for r in sub], dtype=float)
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
            ax.set_xlabel(f"Observation ({unit_txt})")
            ax.set_ylabel(f"Model ({unit_txt})")
            ax.set_title(title_txt)
            ax.grid(True, alpha=0.3)

            metrics = compute_basic_metrics(x, y)
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
            # Use a dedicated bottom axis for colorbar to avoid overlap with x-axis labels/ticks.
            cax = fig.add_axes([0.24, 0.10, 0.52, 0.055])
            cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            cbar.set_label("Water depth (m)")

        fig.suptitle(f"{model_name}: observation vs model", fontsize=11)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.88, wspace=0.22)
        fp = os.path.join(outdir, f"{_sanitize_name(model_name)}_scatter_depth.png")
        plt.savefig(fp, dpi=350)
        plt.close(fig)
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
        _write_csv_rows(raw_file, pair_rows, CTD_PAIR_FIELDS)

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


def _build_color_lookup_from_config():
    cmap = {}
    sch = CONFIG.get("schism", [])
    if isinstance(sch, dict):
        sch = [sch]
    if isinstance(sch, (list, tuple)):
        for item in sch:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            color = str(item.get("color", "")).strip()
            if label:
                cmap[label] = color if color else "b"
    return cmap


def _safe_float_scalar(v, default=np.nan):
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _cmp_from_interp_on_obs_grid(obs_depth, obs_values, model_interp_values):
    obs_depth = np.asarray(obs_depth, dtype=float)
    obs_values = np.asarray(obs_values, dtype=float)
    mod_values = np.asarray(model_interp_values, dtype=float)
    valid = np.isfinite(obs_depth) & np.isfinite(obs_values) & np.isfinite(mod_values)
    if valid.sum() < 2:
        return None
    dep = obs_depth[valid]
    obs = obs_values[valid]
    mod = mod_values[valid]
    return {
        "obs": obs,
        "mod": mod,
        "depth": dep,
        "stat": get_stat(mod, obs),
        "metrics": compute_basic_metrics(obs, mod),
    }


def _resolve_source_npz_paths(source_cfg):
    raw_paths = source_cfg.get("npz_paths")
    if raw_paths is None:
        return []
    if isinstance(raw_paths, (list, tuple, np.ndarray)):
        return [str(p) for p in raw_paths if str(p).strip()]
    return [str(raw_paths)]


def _array_equal_nan_ok(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in {"f", "c"} or b.dtype.kind in {"f", "c"}:
        return bool(np.allclose(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _load_and_merge_source_npz(npz_paths):
    if len(npz_paths) == 0:
        raise ValueError("No source NPZ paths were provided.")
    datasets = []
    for path in npz_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"source npz not found: {path}")
        datasets.append(np.load(path, allow_pickle=True))

    try:
        required = [
            "profile_id",
            "station_id",
            "lon",
            "lat",
            "time_obs",
            "nlev",
            "obs_depth",
            "obs_temp",
            "obs_sal",
            "run_names",
        ]
        for i, ds in enumerate(datasets):
            missing = [k for k in required if k not in ds.files]
            if missing:
                raise KeyError(f"source npz[{i}] missing required keys: {missing}")

        if len(datasets) == 1:
            ds0 = datasets[0]
            return {k: ds0[k] for k in ds0.files}

        run_axis_keys = {
            "run_names",
            "run_dirs",
            "model_start",
            "model_temp",
            "model_sal",
            "model_time",
            "model_lag_hours",
            "model_stack",
            "qc_flag",
            "model_depth_native",
            "model_temp_native",
            "model_sal_native",
            "model_nlev_native",
            "has_interp",
            "has_native",
        }
        obs_axis_keys = {
            "profile_id",
            "station_id",
            "station_name",
            "lon",
            "lat",
            "time_obs",
            "nlev",
            "obs_depth",
            "obs_temp",
            "obs_sal",
            "time_units",
        }

        merged = {}
        ds0 = datasets[0]

        # Keep and verify observation-side arrays once.
        for key in obs_axis_keys:
            if key not in ds0.files:
                continue
            ref = ds0[key]
            for j in range(1, len(datasets)):
                dsj = datasets[j]
                if key not in dsj.files:
                    raise KeyError(f"source npz[{j}] missing key '{key}' required for merge.")
                if not _array_equal_nan_ok(ref, dsj[key]):
                    raise ValueError(f"source npz mismatch for key '{key}' between file[0] and file[{j}].")
            merged[key] = ref

        # Merge run-axis arrays by concatenation.
        union_run_keys = set()
        for ds in datasets:
            for key in ds.files:
                if key in run_axis_keys:
                    union_run_keys.add(key)
        native_varlev_keys = {
            "model_depth_native",
            "model_temp_native",
            "model_sal_native",
        }
        for key in sorted(union_run_keys):
            arrs = []
            for j, ds in enumerate(datasets):
                if key not in ds.files:
                    raise KeyError(f"source npz[{j}] missing run-axis key '{key}' required for merge.")
                arr = np.asarray(ds[key])
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                arrs.append(arr)
            if key in native_varlev_keys:
                ref_ndim = arrs[0].ndim
                ref_prefix = arrs[0].shape[1:-1]
                max_nlev = arrs[0].shape[-1]
                for j, arr in enumerate(arrs[1:], start=1):
                    if arr.ndim != ref_ndim or arr.shape[1:-1] != ref_prefix:
                        raise ValueError(
                            f"run-axis shape mismatch for '{key}': file[0]={arrs[0].shape}, file[{j}]={arr.shape}"
                        )
                    max_nlev = max(max_nlev, int(arr.shape[-1]))

                padded = []
                for arr in arrs:
                    arr_use = arr
                    if arr_use.dtype.kind not in {"f", "c"}:
                        arr_use = arr_use.astype(float)
                    pad_nlev = max_nlev - int(arr_use.shape[-1])
                    if pad_nlev > 0:
                        pad_shape = list(arr_use.shape)
                        pad_shape[-1] = pad_nlev
                        pad_block = np.full(pad_shape, np.nan, dtype=arr_use.dtype)
                        arr_use = np.concatenate([arr_use, pad_block], axis=-1)
                    padded.append(arr_use)
                merged[key] = np.concatenate(padded, axis=0)
            else:
                ref_shape = arrs[0].shape[1:]
                for j, arr in enumerate(arrs[1:], start=1):
                    if arr.shape[1:] != ref_shape:
                        raise ValueError(
                            f"run-axis shape mismatch for '{key}': file[0]={arrs[0].shape}, file[{j}]={arr.shape}"
                        )
                merged[key] = np.concatenate(arrs, axis=0)

        # Keep any other metadata fields from the first file.
        for key in ds0.files:
            if key in merged:
                continue
            merged[key] = ds0[key]
        return merged
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass


def _append_pair_rows(
    pair_rows,
    compares,
    task_name,
    experiment_id,
    model_name,
    station_id,
    station_name,
    profile_id,
    obs_time_txt,
):
    for var in ("temp", "salt"):
        cmp = compares.get(var)
        if cmp is None:
            continue
        obs_vals = np.asarray(cmp["obs"], dtype=float)
        mod_vals = np.asarray(cmp["mod"], dtype=float)
        dep_vals = np.asarray(cmp["depth"], dtype=float)
        for j in range(len(obs_vals)):
            pair_rows.append(
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


def _build_total_summary(total_profiles, summary_chunks, extras=None):
    summary = {
        "mpi_size": SIZE,
        "total_profiles": int(total_profiles),
        "assigned_profiles": int(builtins.sum(s["assigned_profiles"] for s in summary_chunks)),
        "processed_profiles": int(builtins.sum(s["processed_profiles"] for s in summary_chunks)),
        "skipped_profiles": int(builtins.sum(s["skipped_profiles"] for s in summary_chunks)),
        "saved_profile_plots": int(builtins.sum(s["saved_profile_plots"] for s in summary_chunks)),
        "pair_samples": int(builtins.sum(s["pair_samples"] for s in summary_chunks)),
    }
    if extras:
        summary.update(extras)
    return summary


def _finalize_pair_outputs(local_pair_rows, local_summary, total_profiles, output_cfg, chunk_name, summary_extras=None):
    local_summary["pair_samples"] = len(local_pair_rows)
    chunk_dir, _ = _write_pair_chunk(output_cfg["dir"], chunk_name, RANK, local_pair_rows)
    if MPI:
        summary_chunks = COMM.gather(local_summary, root=0)
        COMM.Barrier()
    else:
        summary_chunks = [local_summary]

    if RANK == 0:
        if MPI:
            pair_rows = _collect_pair_chunks(chunk_dir, SIZE)
        else:
            pair_rows = local_pair_rows
        total_summary = _build_total_summary(total_profiles, summary_chunks, extras=summary_extras)
        _write_task_outputs(pair_rows, output_cfg, total_summary)
        _cleanup_pair_chunks(chunk_dir)
        rank_print("done")
    if MPI:
        COMM.Barrier()


def _load_teams_raw_source(teams_cfg):
    teams = loadz(teams_cfg["npz_path"])
    lon = np.asarray(getattr(teams, teams_cfg["lon_name"]))
    lat = np.asarray(getattr(teams, teams_cfg["lat_name"]))
    depth = np.asarray(getattr(teams, teams_cfg["depth_name"]))
    temp = np.asarray(getattr(teams, teams_cfg["temp_name"]))
    salt = np.asarray(getattr(teams, teams_cfg["salt_name"]))
    station_id = np.asarray(getattr(teams, teams_cfg["station_id_name"]))
    station_name = None
    station_name_field = teams_cfg.get("station_name_name")
    if station_name_field and hasattr(teams, station_name_field):
        station_name = np.asarray(getattr(teams, station_name_field))

    time_raw = np.asarray(getattr(teams, teams_cfg["time_name"]))
    time_str = time_raw.astype("datetime64[s]").astype(str)
    obs_time = np.array([datenum(t) for t in time_str], dtype=float)
    uniq_keys, inv, _ = build_profile_groups(station_id, time_raw)
    return {
        "lon": lon,
        "lat": lat,
        "depth": depth,
        "temp": temp,
        "salt": salt,
        "station_id": station_id,
        "station_name": station_name,
        "time_raw": time_raw,
        "obs_time": obs_time,
        "uniq_keys": uniq_keys,
        "inv": inv,
    }


def _run_npz_source(start_t, end_t, region, output_cfg):
    source_cfg = CONFIG.get("source", {})
    npz_paths = _resolve_source_npz_paths(source_cfg)
    if len(npz_paths) == 0:
        raise ValueError("CONFIG['source']['npz_paths'] is required when source.mode='npz'.")
    source = _load_and_merge_source_npz(npz_paths)

    profile_id_all = np.asarray(source["profile_id"]).astype("U")
    station_id_all = np.asarray(source["station_id"]).astype("U")
    station_name_all = np.asarray(source["station_name"]).astype("U") if "station_name" in source else np.asarray([""] * len(profile_id_all)).astype("U")
    lon_all = np.asarray(source["lon"], dtype=float)
    lat_all = np.asarray(source["lat"], dtype=float)
    time_obs_all = np.asarray(source["time_obs"], dtype=float)
    nlev_all = np.asarray(source["nlev"], dtype=int)
    obs_depth_all = np.asarray(source["obs_depth"], dtype=float)
    obs_temp_all = np.asarray(source["obs_temp"], dtype=float)
    obs_sal_all = np.asarray(source["obs_sal"], dtype=float)

    run_names = np.asarray(source["run_names"]).astype("U")
    nrun = len(run_names)
    if nrun == 0:
        raise RuntimeError("No runs found in source npz.")

    model_time_all = np.asarray(source["model_time"], dtype=float) if "model_time" in source else None
    model_temp_interp_all = np.asarray(source["model_temp"], dtype=float) if "model_temp" in source else None
    model_sal_interp_all = np.asarray(source["model_sal"], dtype=float) if "model_sal" in source else None
    model_depth_native_all = np.asarray(source["model_depth_native"], dtype=float) if "model_depth_native" in source else None
    model_temp_native_all = np.asarray(source["model_temp_native"], dtype=float) if "model_temp_native" in source else None
    model_sal_native_all = np.asarray(source["model_sal_native"], dtype=float) if "model_sal_native" in source else None
    model_nlev_native_all = np.asarray(source["model_nlev_native"], dtype=int) if "model_nlev_native" in source else None
    has_interp_all = np.asarray(source["has_interp"], dtype=int) if "has_interp" in source else np.ones(nrun, dtype=int)
    has_native_all = np.asarray(source["has_native"], dtype=int) if "has_native" in source else np.zeros(nrun, dtype=int)

    color_lookup = _build_color_lookup_from_config()
    default_colors = ["b", "g", "m", "c", "y", "k", "tab:orange", "tab:brown"]
    plot_depth_mode = str(source_cfg.get("plot_depth_mode", "native")).strip().lower()
    plot_runtime = _prepare_plot_runtime(output_cfg)

    total_profiles = len(profile_id_all)
    local_indices = [i for i in range(total_profiles) if i % SIZE == RANK]
    _report_profile_assignment("npz source", total_profiles, local_indices)

    compare_cfg = CONFIG.get("compare", {})
    station_ids = compare_cfg.get("station_ids")
    station_names = compare_cfg.get("station_names")
    if station_ids is not None:
        station_ids = set(str(s) for s in station_ids)
    if station_names is not None:
        station_names = set(str(s) for s in station_names)

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

    for pidx in local_indices:
        profile_id = str(profile_id_all[pidx]).strip()
        station_id = str(station_id_all[pidx]).strip()
        station_name = str(station_name_all[pidx]).strip()
        lon_pt = _safe_float_scalar(lon_all[pidx])
        lat_pt = _safe_float_scalar(lat_all[pidx])
        stime = _safe_float_scalar(time_obs_all[pidx])

        if station_ids is not None and station_id not in station_ids:
            local_summary["skipped_profiles"] += 1
            continue
        if station_names is not None and station_name not in station_names:
            local_summary["skipped_profiles"] += 1
            continue
        if not np.isfinite(stime) or stime < start_t or stime > end_t:
            local_summary["skipped_profiles"] += 1
            continue
        if (not np.isfinite(lon_pt)) or (not np.isfinite(lat_pt)) or (not point_in_region(region, lon_pt, lat_pt)):
            local_summary["skipped_profiles"] += 1
            continue

        nz = int(nlev_all[pidx])
        if nz < 2:
            local_summary["skipped_profiles"] += 1
            continue
        obs_depth0 = np.asarray(obs_depth_all[pidx, :nz], dtype=float)
        obs_temp0 = np.asarray(obs_temp_all[pidx, :nz], dtype=float)
        obs_sal0 = np.asarray(obs_sal_all[pidx, :nz], dtype=float)
        valid_obs = np.isfinite(obs_depth0) & np.isfinite(obs_temp0) & np.isfinite(obs_sal0)
        if valid_obs.sum() < 4:
            local_summary["skipped_profiles"] += 1
            continue
        keep_idx = np.where(valid_obs)[0]
        obs_depth = obs_depth0[keep_idx]
        obs_temp = obs_temp0[keep_idx]
        obs_sal = obs_sal0[keep_idx]
        o = np.argsort(obs_depth)
        keep_idx = keep_idx[o]
        obs_depth = obs_depth[o]
        obs_temp = obs_temp[o]
        obs_sal = obs_sal[o]

        models = []
        for ir in range(nrun):
            model_name = str(run_names[ir]).strip() or f"run_{ir + 1}"
            model_color = color_lookup.get(model_name, default_colors[ir % len(default_colors)])
            model_time = stime
            if model_time_all is not None and model_time_all.ndim == 2:
                model_time = _safe_float_scalar(model_time_all[ir, pidx], default=stime)

            has_interp = bool(ir < len(has_interp_all) and int(has_interp_all[ir]) == 1 and model_temp_interp_all is not None and model_sal_interp_all is not None)
            has_native = bool(ir < len(has_native_all) and int(has_native_all[ir]) == 1 and model_depth_native_all is not None and model_temp_native_all is not None and model_sal_native_all is not None and model_nlev_native_all is not None)

            cmp_temp = None
            cmp_salt = None

            # Metrics always on observation grid.
            if has_interp:
                mt_i = np.asarray(model_temp_interp_all[ir, pidx, :nz], dtype=float)[keep_idx]
                ms_i = np.asarray(model_sal_interp_all[ir, pidx, :nz], dtype=float)[keep_idx]
                cmp_temp = _cmp_from_interp_on_obs_grid(obs_depth, obs_temp, mt_i)
                cmp_salt = _cmp_from_interp_on_obs_grid(obs_depth, obs_sal, ms_i)
            if (cmp_temp is None and cmp_salt is None) and has_native:
                nn = int(model_nlev_native_all[ir, pidx])
                if nn >= 2:
                    md = np.asarray(model_depth_native_all[ir, pidx, :nn], dtype=float)
                    mt = np.asarray(model_temp_native_all[ir, pidx, :nn], dtype=float)
                    ms = np.asarray(model_sal_native_all[ir, pidx, :nn], dtype=float)
                    cmp_temp = interpolate_model_to_obs(md, mt, obs_depth, obs_temp)
                    cmp_salt = interpolate_model_to_obs(md, ms, obs_depth, obs_sal)

            if cmp_temp is None and cmp_salt is None:
                continue

            # Plot mode can still use native profiles.
            p_depth = obs_depth
            p_temp = np.full(obs_depth.shape, np.nan, dtype=float)
            p_salt = np.full(obs_depth.shape, np.nan, dtype=float)
            if plot_depth_mode == "native" and has_native:
                nn = int(model_nlev_native_all[ir, pidx])
                if nn >= 2:
                    p_depth = np.asarray(model_depth_native_all[ir, pidx, :nn], dtype=float)
                    p_temp = np.asarray(model_temp_native_all[ir, pidx, :nn], dtype=float)
                    p_salt = np.asarray(model_sal_native_all[ir, pidx, :nn], dtype=float)
            elif has_interp:
                p_temp = np.asarray(model_temp_interp_all[ir, pidx, :nz], dtype=float)[keep_idx]
                p_salt = np.asarray(model_sal_interp_all[ir, pidx, :nz], dtype=float)[keep_idx]

            models.append(
                {
                    "name": model_name,
                    "color": model_color,
                    "depth": p_depth,
                    "temp": p_temp,
                    "salt": p_salt,
                    "time": model_time,
                    "stats": {
                        "temp": cmp_temp["stat"] if cmp_temp is not None else None,
                        "salt": cmp_salt["stat"] if cmp_salt is not None else None,
                    },
                    "compare": {"temp": cmp_temp, "salt": cmp_salt},
                }
            )

        if len(models) == 0:
            local_summary["skipped_profiles"] += 1
            continue

        local_summary["processed_profiles"] += 1
        obs_time_txt = num2date(stime).strftime("%Y-%m-%d %H:%M:%S")
        for model in models:
            model_name = str(model.get("name", "model"))
            compares = model.get("compare", {})
            _append_pair_rows(
                local_pair_rows,
                compares,
                task_name,
                experiment_id,
                model_name,
                station_id,
                station_name,
                profile_id,
                obs_time_txt,
            )

        if _should_plot_profile(pidx, local_summary["saved_profile_plots"], plot_runtime):
            obs_plot = {"depth": obs_depth, "temp": obs_temp, "salt": obs_sal}
            meta = {
                "lon": lon_pt,
                "lat": lat_pt,
                "lon_array": np.array([lon_pt]),
                "lat_array": np.array([lat_pt]),
                "obs_time": stime,
                "profile_id": profile_id,
                "gm_lon": None,
                "gm_lat": None,
            }
            plot_profiles(meta, region, obs_plot, models, CONFIG["plot"], output_cfg["dir"], plot_runtime=plot_runtime)
            local_summary["saved_profile_plots"] += 1

    _finalize_pair_outputs(
        local_pair_rows,
        local_summary,
        total_profiles,
        output_cfg,
        ".ctd_pair_chunks_npz",
        summary_extras={
            "source_mode": "npz",
            "source_npz": npz_paths[0] if len(npz_paths) == 1 else npz_paths,
            "source_npz_count": int(len(npz_paths)),
            "plot_depth_mode": plot_depth_mode,
        },
    )


def _run_raw_source(start_t, end_t, region, output_cfg, plot_runtime):
    sch_cfg = prepare_schism(CONFIG["schism"])
    gm_cfg = prepare_global_model(CONFIG["global_model"], start_t, end_t)
    sch_list = sch_cfg if isinstance(sch_cfg, list) else ([sch_cfg] if sch_cfg is not None else [])
    active_models = [cfg for cfg in ([gm_cfg] + sch_list) if cfg is not None]
    if len(active_models) == 0:
        rank_print("No models selected for comparison. Enable at least one model in CONFIG.")
        return

    teams = _load_teams_raw_source(CONFIG["teams"])
    lon = teams["lon"]
    lat = teams["lat"]
    depth = teams["depth"]
    temp = teams["temp"]
    salt = teams["salt"]
    st_id = teams["station_id"]
    st_name = teams["station_name"]
    obs_time = teams["obs_time"]
    uniq_keys = teams["uniq_keys"]
    inv = teams["inv"]

    total_profiles = len(uniq_keys)
    if total_profiles == 0:
        rank_print("No TEAMS profiles found.")
        return

    local_indices = [i for i in range(total_profiles) if i % SIZE == RANK]
    _report_profile_assignment("raw source", total_profiles, local_indices)

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
            _append_pair_rows(
                local_pair_rows,
                compares,
                task_name,
                experiment_id,
                model_name,
                station_id,
                station_name,
                profile_id,
                obs_time_txt,
            )

        if _should_plot_profile(profile_idx, local_summary["saved_profile_plots"], plot_runtime):
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
            plot_profiles(meta, region, obs_plot, models, CONFIG["plot"], output_cfg["dir"], plot_runtime=plot_runtime)
            local_summary["saved_profile_plots"] += 1

    _finalize_pair_outputs(
        local_pair_rows,
        local_summary,
        total_profiles,
        output_cfg,
        ".ctd_pair_chunks_raw",
    )


def main():
    start_t, end_t = parse_date_range(CONFIG["date_range"])
    region = setup_region(CONFIG["region"])
    output_cfg = CONFIG.get("output", {})
    plot_runtime = _prepare_plot_runtime(output_cfg)
    source_cfg = CONFIG.get("source", {})
    source_mode = str(source_cfg.get("mode", "raw")).strip().lower()

    if RANK == 0:
        ensure_output_dir(output_cfg["dir"])
        with open(os.path.join(output_cfg["dir"], "ctd_config_used.json"), "w", encoding="utf-8") as f:
            json.dump(CANONICAL_CONFIG, f, indent=2)
    if MPI:
        COMM.Barrier()

    if source_mode == "npz":
        _run_npz_source(start_t, end_t, region, output_cfg)
        return

    _run_raw_source(start_t, end_t, region, output_cfg, plot_runtime)


if __name__ == "__main__":
    cli_args = _parse_args()
    _apply_runtime_overrides(cli_args)
    main()
    sys.exit()
