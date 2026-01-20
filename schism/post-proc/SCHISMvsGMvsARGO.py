import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pylib import *
import numpy as np
import numpy.ma as ma
import sys

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
        "start": (2022, 2, 1),
        "end": (2022, 4, 1),
    },
    "region": {
        "shapefile": "hgrid_bnd.shp",  # set to None to skip shapefile filtering
        "use_shapefile": True,
        # (lon_min, lon_max, lat_min, lat_max); set to None to skip bounding-box filtering
        "subset_bbox": None,#(80,100,22,30),
    },
    "argo": {
        "dir": "/storage/coda1/p-ed70/0/kpark350/dataset/ARGO/NP/2022/",
        "reference_time": (1950, 1, 1),
    },
    "schism": {
        "enabled": True,
        "label": "SCHISM",
        "color": "b",
        "run_dir": "/storage/coda1/p-ed70/0/kpark350/Projects/EastAsia/run/RUN02a",
        "variables": ["temp", "salt"],
        "refdate": datenum("2022-1-2"),
        "stack_range": (1, 1830),
        "stack_step": 1 / 24,
        "lon_mode": "360",  # set to "360" if SCHISM output uses 0-360 longitudes
    },
    "global_model": {
        "enabled": True,
        "label": "CMEMS",
        "color": "k",
        "data_dir": "/storage/coda1/p-ed70/0/kpark350/dataset/CMEMS/EastAsia/",
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
    },
    "output": {
        "dir": "./argo_schism/each_profile/SCHISMvsCMEMSvsARGO/RUN02a",
    },
    "plot": {
        "linewidth": 1.5,
        "font_size": 7,
        "obs_color": "r",
    },
}


#######################################################################################################################
# control font size in plot
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
#######################################################################################################################

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
        lon_in_box = False
        if lon_min <= lon_max:
            lon_in_box = lon_min <= lon <= lon_max
        else:  # dateline-crossing box
            lon_in_box = lon >= lon_min or lon <= lon_max
        if not (lon_in_box and lat_min <= lat <= lat_max):
            return False
    return True


def prepare_argo_files(argo_cfg):
    dir_argo = argo_cfg["dir"]
    files = array([f for f in os.listdir(dir_argo) if f.endswith(".nc")])
    if len(files) == 0:
        return array([]), array([])
    mti = array([f.replace(".nc", "") for f in files])
    sind = argsort(mti)
    return files[sind], mti[sind]


def sanitize_masked(arr):
    data = array(arr, dtype=float)
    mask = getattr(arr, "mask", None)
    if mask is not None:
        data[mask == 1] = NaN
    return data


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
    model_depth, model_values = drop_nan_pairs(model_depth, model_values)
    if len(model_depth) < 2 or len(obs_depth) < 2:
        return None
    interp_fn = interpolate.interp1d(model_depth, model_values, bounds_error=False, fill_value=NaN)
    model_interp = interp_fn(obs_depth)
    valid = (~isnan(model_interp)) & (~isnan(obs_values))
    if valid.sum() < 2:
        return None
    return get_stat(model_interp[valid], obs_values[valid])


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
        "cache_index": None,
        "cache_ds": None,
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
    return cfg["cache_ds"]


def extract_global_profile(cfg, stime, lon, lat):
    if cfg is None:
        return None
    times = cfg["times"]
    idx = abs(times - stime).argmin()
    ds = load_global_dataset(cfg, idx)
    vars_ = cfg["variables"]

    lon_var = ds.variables[vars_["lon"]]
    lat_var = ds.variables[vars_["lat"]]
    depth_var = ds.variables[vars_["depth"]]

    lon_data = np.asarray(lon_var[:], dtype=float).squeeze()
    lat_data = np.asarray(lat_var[:], dtype=float).squeeze()
    depth_data = np.asarray(depth_var[:], dtype=float).squeeze()

    if lon_data.ndim > 2 or lat_data.ndim > 2:
        return None

    lon_mode = str(cfg.get("lon_mode", "auto")).lower()
    lon_canonical = normalize_longitudes(lon, CANONICAL_LON_MODE)

    def convert_lon(arr, mode):
        if mode == "360":
            return normalize_longitudes(arr, "360"), normalize_longitudes(lon_canonical, "360")
        if mode in {"180", "canonical"}:
            return normalize_longitudes(arr, "180"), normalize_longitudes(lon_canonical, "180")
        return arr.astype(float), lon_canonical

    if lon_data.ndim == 1 and lat_data.ndim == 1:
        if lon_mode == "auto":
            lon_norm = lon_data.astype(float)
            target_lon = lon_canonical
            if np.any(lon_norm > 180):
                lon_norm, target_lon = convert_lon(lon_norm, "180")
        else:
            lon_norm, target_lon = convert_lon(lon_data, lon_mode)
        lon_idx = int(np.abs(lon_norm - target_lon).argmin())
        lat_idx = int(np.abs(lat_data - lat).argmin())
    else:
        if lon_mode == "auto":
            lon_norm = lon_data.astype(float)
            target_lon = lon_canonical
            if np.any(lon_norm > 180):
                lon_norm, target_lon = convert_lon(lon_norm, "180")
        else:
            lon_norm, target_lon = convert_lon(lon_data, lon_mode)
        lon_grid, lat_grid = np.broadcast_arrays(lon_norm, lat_data)
        dist = (lat_grid - lat) ** 2 + (lon_grid - target_lon) ** 2
        llidx = unravel_index(dist.argmin(), dist.shape)
        lat_idx, lon_idx = int(llidx[0]), int(llidx[1])

    def extract_profile(var):
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
            lat_idx_clipped = int(np.clip(lat_idx, 0, var.shape[lat_axis] - 1))
            slices[lat_axis] = lat_idx_clipped
        if lon_axis is not None:
            lon_idx_clipped = int(np.clip(lon_idx, 0, var.shape[lon_axis] - 1))
            slices[lon_axis] = lon_idx_clipped

        data = var[tuple(slices)]
        if ma.isMaskedArray(data):
            data = data.filled(np.nan)
        data = np.asarray(data, dtype=float)
        data = np.squeeze(data)
        if data.ndim == 0:
            data = data.reshape(1)
        return data

    temp_profile = extract_profile(ds.variables[vars_["temp"]])
    salt_profile = extract_profile(ds.variables[vars_["salt"]])

    fill_value = cfg.get("fill_value")
    if fill_value is not None:
        temp_profile[np.isclose(temp_profile, fill_value, atol=1e-6, rtol=0)] = NaN
        salt_profile[np.isclose(salt_profile, fill_value, atol=1e-6, rtol=0)] = NaN

    temp_profile[np.abs(temp_profile) > 1e30] = NaN
    salt_profile[np.abs(salt_profile) > 1e30] = NaN

    return {
        "name": cfg["label"],
        "color": cfg["color"],
        "depth": depth_data,
        "temp": temp_profile,
        "salt": salt_profile,
        "time": cfg["times"][idx],
    }


def prepare_schism(cfg):
    if not cfg.get("enabled", False):
        return None
    cfg = cfg.copy()
    stack_start, stack_end = cfg["stack_range"]
    stack_step = cfg.get("stack_step", 1)
    stacks = arange(stack_start, stack_end + stack_step, stack_step)
    cfg["stacks"] = stacks
    cfg["mti"] = stacks - 1 + cfg["refdate"]
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
    alon = meta["lon_array"]
    alat = meta["lat_array"]
    stime = meta["obs_time"]

    obs_time_str = num2date(stime).strftime("%Y-%m-%d %H:%M:%S")
    model_time_labels = ", ".join([format_time_label(m["name"], m["time"]) for m in models])

    depth_temp_obs, temp_obs = drop_nan_pairs(obs_plot["depth"], obs_plot["temp"])
    depth_salt_obs, salt_obs = drop_nan_pairs(obs_plot["depth"], obs_plot["salt"])
    if len(depth_temp_obs) == 0 or len(depth_salt_obs) == 0:
        return

    figure(1, figsize=[7.2, 3.5])
    clf()
    subplot(1, 2, 1)
    plot(alon, alat, f"{obs_color}+", markersize=10, label="Obs")
    for model in models:
        plot([], [], model["color"], lw=linewidth, label=model["name"])
    if region["px"] is not None and region["py"] is not None:
        plot(region["px"], region["py"], "k")
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

    figure(1, figsize=[7.2, 3.5])
    clf()
    subplot(1, 2, 1)
    plot(alon, alat, f"{obs_color}+", markersize=10, label="Obs")
    for model in models:
        plot([], [], model["color"], lw=linewidth, label=model["name"])
    if region["px"] is not None and region["py"] is not None:
        plot(region["px"], region["py"], "k")
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


def main():
    start_t, end_t = parse_date_range(CONFIG["date_range"])
    region = setup_region(CONFIG["region"])
    argo_files, argo_ids = prepare_argo_files(CONFIG["argo"])
    ref_time_argo = datenum(*CONFIG["argo"]["reference_time"])

    gm_cfg = prepare_global_model(CONFIG["global_model"], start_t, end_t)
    sch_cfg = prepare_schism(CONFIG["schism"])
    active_models = [cfg for cfg in (gm_cfg, sch_cfg) if cfg is not None]
    if len(active_models) == 0:
        rank_print("No models selected for comparison. Enable at least one model in CONFIG.")
        return

    total_profiles = len(argo_files)
    if total_profiles == 0:
        if RANK == 0:
            rank_print("No Argo profiles found in directory.")
        return

    if RANK == 0:
        ensure_output_dir(CONFIG["output"]["dir"])
    if MPI:
        COMM.Barrier()

    dir_argo = CONFIG["argo"]["dir"]

    local_indices = [i for i in range(total_profiles) if i % SIZE == RANK]
    local_total = len(local_indices)
    rank_print(f"Assigned {local_total} of {total_profiles} profiles.")
    if local_total == 0:
        rank_print("No profiles assigned to this rank.")

    for local_pos, global_idx in enumerate(local_indices, 1):
        fname = argo_files[global_idx]
        profile_id = argo_ids[global_idx]
        rank_print(
            f"Working on {local_pos}/{local_total} (global {global_idx + 1}/{total_profiles})"
        )
        S = ReadNC(os.path.join(dir_argo, fname))
        stimes = S.JULD.val.data + ref_time_argo
        depth = S.PRES_ADJUSTED.val * 1.01998
        temp = S.TEMP_ADJUSTED.val
        salt = S.PSAL_ADJUSTED.val
        alon_raw = S.LONGITUDE.val
        alon = normalize_longitudes(alon_raw, CANONICAL_LON_MODE)
        alat = S.LATITUDE.val

        for nn, stime in enumerate(stimes):
            lon_pt = alon[nn]
            lat_pt = alat[nn]
            if stime < start_t or stime > end_t:
                log_skip("outside target window", profile_id, stime, lon_pt, lat_pt)
                continue
            if not point_in_region(region, lon_pt, lat_pt):
                log_skip("outside configured region", profile_id, stime, lon_pt, lat_pt)
                continue

            obs_depth = sanitize_masked(depth[nn, :])
            obs_temp = sanitize_masked(temp[nn, :])
            obs_salt = sanitize_masked(salt[nn, :])

            valid = ~isnan(obs_depth)
            if valid.sum() < 4:
                log_skip("insufficient depth samples", profile_id, stime, lon_pt, lat_pt)
                continue
            obs_depth_clean = obs_depth[valid]
            obs_temp_clean = obs_temp[valid]
            obs_salt_clean = obs_salt[valid]
            if len(obs_depth_clean) == 0:
                log_skip("no valid observation depths", profile_id, stime, lon_pt, lat_pt)
                continue

            models = []
            if gm_cfg is not None:
                gm_profile = extract_global_profile(gm_cfg, stime, lon_pt, lat_pt)
                if gm_profile is not None:
                    obs_subset = select_obs_within_model_range(
                        obs_depth_clean, obs_temp_clean, obs_salt_clean, gm_profile["depth"]
                    )
                    if obs_subset is not None:
                        odi, otpi, osi = obs_subset
                        temp_stat = compute_model_stats(gm_profile["depth"], gm_profile["temp"], odi, otpi)
                        salt_stat = compute_model_stats(gm_profile["depth"], gm_profile["salt"], odi, osi)
                        gm_profile["stats"] = {"temp": temp_stat, "salt": salt_stat}
                        models.append(gm_profile)
                    else:
                        log_skip("global model depth range mismatch", profile_id, stime, lon_pt, lat_pt)
                else:
                    log_skip("global model profile missing", profile_id, stime, lon_pt, lat_pt)
            if sch_cfg is not None:
                sch_profile = extract_schism_profile(sch_cfg, stime, lon_pt, lat_pt)
                if sch_profile is not None:
                    obs_subset = select_obs_within_model_range(
                        obs_depth_clean, obs_temp_clean, obs_salt_clean, sch_profile["depth"]
                    )
                    if obs_subset is not None:
                        odi, otpi, osi = obs_subset
                        temp_stat = compute_model_stats(sch_profile["depth"], sch_profile["temp"], odi, otpi)
                        salt_stat = compute_model_stats(sch_profile["depth"], sch_profile["salt"], odi, osi)
                        sch_profile["stats"] = {"temp": temp_stat, "salt": salt_stat}
                        models.append(sch_profile)
                    else:
                        log_skip("SCHISM depth range mismatch", profile_id, stime, lon_pt, lat_pt)
                else:
                    log_skip("SCHISM profile missing", profile_id, stime, lon_pt, lat_pt)

            if len(models) == 0:
                log_skip("no models available for comparison", profile_id, stime, lon_pt, lat_pt)
                continue

            # ensure stats dictionary exists even if missing
            for model in models:
                model.setdefault("stats", {"temp": None, "salt": None})

            obs_plot = {
                "depth": obs_depth_clean,
                "temp": obs_temp_clean,
                "salt": obs_salt_clean,
            }
            meta = {
                "lon": lon_pt,
                "lat": lat_pt,
                "lon_array": alon,
                "lat_array": alat,
                "obs_time": stime,
                "profile_id": profile_id,
            }
            plot_profiles(meta, region, obs_plot, models, CONFIG["plot"], CONFIG["output"]["dir"])

    if MPI:
        COMM.Barrier()
    if RANK == 0:
        rank_print("done")


if __name__ == "__main__":
    main()
    sys.exit()
