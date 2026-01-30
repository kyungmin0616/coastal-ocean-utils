#!/usr/bin/env python3
"""
Final CFSv2 → SCHISM sflux generator (subset-first, interpolate only if grids differ).

What this script reads and writes
---------------------------------
Input files (CFSv2 monthly NetCDF, per year folder):
  {CFS_ROOT}/YYYY/wnd10m.cdas1.YYYYMM.grb2.nc  -> vars: u, v
  {CFS_ROOT}/YYYY/tmp2m.cdas1.YYYYMM.grb2.nc   -> var: stmp
  {CFS_ROOT}/YYYY/q2m.cdas1.YYYYMM.grb2.nc     -> var: spfh
  {CFS_ROOT}/YYYY/prmsl.cdas1.YYYYMM.grb2.nc   -> var: prmsl
  {CFS_ROOT}/YYYY/dlwsfc.cdas1.YYYYMM.grb2.nc  -> var: dlwrf
  {CFS_ROOT}/YYYY/dswsfc.cdas1.YYYYMM.grb2.nc  -> var: dswrf
  {CFS_ROOT}/YYYY/prate.cdas1.YYYYMM.grb2.nc   -> var: prate

Output files (SCHISM sflux):
  sflux_air_1.NNNN.nc  -> prmsl, spfh, stmp, uwind, vwind
  sflux_rad_1.NNNN.nc  -> dlwrf, dswrf
  sflux_prc_1.NNNN.nc  -> prate

Config dictionary (CONFIG) overview
-----------------------------------
  CFS_ROOT    : root folder with year subfolders of CFSv2 files.
  START/END   : time window (inclusive) in "YYYY-MM-DD HH:MM:SS".
  BBOX        : lon/lat subset [lon_min, lon_max, lat_min, lat_max].
  OUTDIR      : output folder for sflux files.
  STACK_HOURS : hours per output file (e.g., 24 -> daily files if hourly).
  INDEX_PAD   : file index zero padding (0 -> no padding, 4 -> 0001).
  LON_SYSTEM  : "360" for [0,360), "180" for [-180,180).
  OUTPUT_LON_SYSTEM : output lon system for sflux ("180" for WGS84).
  OVERWRITE   : overwrite existing sflux files if True.
  USE_MPI     : enable MPI ranks (requires mpi4py, use mpirun/srun).
  STREAM_MODE : "all" (single run) or "month" (process month-by-month).
  MISSING_POLICY : "error", "fill", or "skip" when a variable is missing.
  MISSING_FILL   : fill value when MISSING_POLICY="fill".
  PROGRESS    : print per-file open progress (fallback mode).
  PROGRESS_EVERY : print every N files when PROGRESS is True.
  PROGRESS_RANK0_ONLY : only rank 0 prints per-file progress.
  PATH_GLOBS  : glob patterns for each CFSv2 variable file.

Examples
--------
1) CLI-only (overrides CONFIG):
  python make_sflux_cfs2_final.py \
    --cfs-root ./CFSv2 \
    --start "2017-01-01 00:00:00" --end "2017-12-31 23:59:59" \
    --bbox 140 142 37 39 \
    --outdir ./Sendai --stack-hours 24 --lon-system 360

2) MPI run (8 ranks):
  mpirun -np 8 python make_sflux_cfs2_final.py --use-mpi \
    --cfs-root ./CFSv2 --start "2017-01-01 00:00:00" --end "2017-12-31 23:59:59" \
    --bbox 140 142 37 39 --outdir ./Sendai --stack-hours 24 --lon-system 360

3) Use a JSON config file:
  python make_sflux_cfs2_final.py --config config_cfs.json

Flags (CLI) overview
--------------------
  --cfs-root         Root directory containing yearly CFSv2 folders.
  --start/--end      Time window (YYYY-MM-DD HH:MM:SS).
  --bbox             lon_min lon_max lat_min lat_max.
  --outdir           Output directory for sflux files.
  --stack-hours      Hours per stack file.
  --lon-system       "360" or "180" longitude system.
  --output-lon-system Output lon system for sflux ("180" for WGS84).
  --index-pad        Zero padding width for file index.
  --overwrite        Overwrite existing outputs.
  --use-mpi          Enable MPI ranks (requires mpirun/srun).
  --stream-monthly   Process and write outputs month-by-month.
  --missing-policy   error | fill | skip.
  --missing-fill     Fill value when --missing-policy=fill.
  --progress         Print per-file open progress (fallback mode).
  --progress-every   Print every N files when --progress is set.
  --progress-all     Print progress from all ranks (default: rank 0 only).
  --config           JSON config file for overrides.
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

try:
    import dask  # noqa: F401
    _HAS_DASK = True
except Exception:
    _HAS_DASK = False

# ==========================
# Config (defaults)
# ==========================
CONFIG = dict(
    CFS_ROOT="./CFSv2",
    START="2017-01-01 00:00:00",
    END="2017-12-31 23:59:59",
    BBOX=dict(lon_min=140.0, lon_max=142.0, lat_min=37.0, lat_max=39.0),
    OUTDIR="./sflux_out",
    STACK_HOURS=24,
    INDEX_PAD=0,       # 0 -> no zero padding (e.g., sflux_air_1.3.nc); 4 -> sflux_air_1.0003.nc
    LON_SYSTEM="360",   # "360" for [0,360), "180" for [-180,180)
    OUTPUT_LON_SYSTEM="180",  # output sflux lon system ("180" for WGS84)
    OVERWRITE=False,
    USE_MPI=False,
    STREAM_MODE="month",  # "all" or "month"
    MISSING_POLICY="error",  # "error", "fill", or "skip"
    MISSING_FILL=np.nan,     # used when MISSING_POLICY="fill"
    PROGRESS=True,          # print per-file open progress
    PROGRESS_EVERY=10,       # print every N files when PROGRESS is True
    PROGRESS_RANK0_ONLY=True,  # only rank 0 prints per-file progress
    PATH_GLOBS=dict(
        wnd10m=["**/wnd10m.cdas1.*.grb2.nc"],
        tmp2m=["**/tmp2m.cdas1.*.grb2.nc"],
        q2m=["**/q2m.cdas1.*.grb2.nc"],
        prmsl=["**/prmsl.cdas1.*.grb2.nc"],
        prate=["**/prate.cdas1.*.grb2.nc"],
        dswsfc=["**/dswsfc.cdas1.*.grb2.nc"],
        dlwsfc=["**/dlwsfc.cdas1.*.grb2.nc"],
    ),
)

# CFSv2 variable mapping: (file_key, var_name_in_file)
AIR_MAP = dict(
    uwind=("wnd10m", "u"),
    vwind=("wnd10m", "v"),
    prmsl=("prmsl", "prmsl"),
    spfh=("q2m", "spfh"),
    stmp=("tmp2m", "stmp"),
)
RAD_MAP = dict(
    dlwrf=("dlwsfc", "dlwrf"),
    dswrf=("dswsfc", "dswrf"),
)
PRC_MAP = dict(
    prate=("prate", "prate"),
)


# ==========================
# MPI (optional)
# ==========================
COMM = None
RANK = 0
SIZE = 1
try:
    from mpi4py import MPI  # type: ignore
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1


def log(msg):
    prefix = f"[rank {RANK}/{SIZE}] " if SIZE > 1 else ""
    print(prefix + str(msg), flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _find_files(root, patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(files))


def _month_range(start, end):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    y, m = s.year, s.month
    out = []
    while (y, m) <= (e.year, e.month):
        out.append(y * 100 + m)
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return set(out)


def _month_start_end(yyyymm):
    start = pd.Timestamp(f"{yyyymm}01 00:00:00")
    end = (start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


def _filter_files_by_month(files, start, end):
    allowed = _month_range(start, end)
    out = []
    for fp in files:
        m = re.search(r"\.(\d{6})\.grb2\.nc$", fp)
        if m is None:
            out.append(fp)
            continue
        yyyymm = int(m.group(1))
        if yyyymm in allowed:
            out.append(fp)
    return out


def _normalize_lon(lon, system):
    lon = np.asarray(lon, dtype=float)
    if system == "360":
        return lon % 360.0
    if system == "180":
        return ((lon + 180.0) % 360.0) - 180.0
    raise ValueError("LON_SYSTEM must be '360' or '180'")


def _apply_output_lon(ds, output_system):
    if output_system is None:
        return ds
    if "lon" not in ds.coords:
        return ds
    lon = _normalize_lon(ds.lon, output_system)
    return ds.assign_coords(lon=lon).sortby("lon")


def _reorder_lon(ds, lon_order):
    if "lon" not in ds.coords:
        return ds
    return ds.isel(lon=lon_order)


def _apply_output_lon_order(target_ds, groups, output_system):
    if output_system is None:
        return target_ds, groups
    target_ds = _apply_output_lon(target_ds, output_system)
    lon_vals = target_ds.lon.values
    lon_order = np.argsort(lon_vals)
    target_ds = _reorder_lon(target_ds, lon_order)
    new_groups = {}
    for gname, group in groups.items():
        new_group = {}
        for name, da in group.items():
            if da is None:
                new_group[name] = None
                continue
            da = _apply_output_lon(da, output_system)
            da = _reorder_lon(da, lon_order)
            new_group[name] = da
        new_groups[gname] = new_group
    return target_ds, new_groups


def _normalize_lon_lat(ds, lon_system):
    rename = {}
    if "lon" not in ds.coords:
        for cand in ("longitude", "x"):
            if cand in ds.coords:
                rename[cand] = "lon"
                break
    if "lat" not in ds.coords:
        for cand in ("latitude", "y"):
            if cand in ds.coords:
                rename[cand] = "lat"
                break
    if rename:
        ds = ds.rename(rename)
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("Dataset missing lon/lat coordinates.")
    ds = ds.assign_coords(lon=_normalize_lon(ds.lon, lon_system)).sortby("lon")
    if not np.all(np.diff(ds.lon.values) > 0):
        ds = ds.sortby("lon")
    if not np.all(np.diff(ds.lat.values) > 0):
        ds = ds.sortby("lat")
    return ds


def _normalize_bbox_lon(lon_min, lon_max, system):
    lon_min_n = float(_normalize_lon(lon_min, system))
    lon_max_n = float(_normalize_lon(lon_max, system))
    wraps = lon_max_n < lon_min_n
    return lon_min_n, lon_max_n, wraps


def _subset_lon_wrap(ds, lon_min, lon_max, lon_system):
    lon_min_n, lon_max_n, wraps = _normalize_bbox_lon(lon_min, lon_max, lon_system)
    if not wraps:
        return ds.sel(lon=slice(lon_min_n, lon_max_n))
    right = ds.sel(lon=slice(lon_min_n, ds.lon.max().item()))
    left = ds.sel(lon=slice(ds.lon.min().item(), lon_max_n))
    return xr.concat([right, left], dim="lon").sortby("lon")


def _subset_bbox(ds, bbox, lon_system):
    ds = _subset_lon_wrap(ds, bbox["lon_min"], bbox["lon_max"], lon_system)
    return ds.sel(lat=slice(bbox["lat_min"], bbox["lat_max"]))


def _ensure_time_sorted_unique(ds):
    if "time" not in ds.coords:
        return ds
    t = pd.to_datetime(ds["time"].values)
    order = np.argsort(t)
    t_sorted = t[order]
    unique_mask = ~pd.Index(t_sorted).duplicated(keep="first")
    ds = ds.isel(time=order)
    ds = ds.isel(time=unique_mask)
    ds = ds.assign_coords(time=t_sorted[unique_mask])
    return ds.sortby("time")


def _open_dataset(paths, lon_system, start, end, bbox, progress=False, progress_every=10,
                  progress_rank0_only=True):
    if not paths:
        raise FileNotFoundError("No input files found.")
    try:
        if _HAS_DASK:
            ds = xr.open_mfdataset(paths, combine="by_coords", decode_times=True, parallel=True)
        else:
            raise ModuleNotFoundError("dask")
    except (ModuleNotFoundError, ValueError) as e:
        if "dask" not in str(e):
            raise
        # Fallback when dask is not available: subset per-file, then concat.
        datasets = []
        total = len(paths)
        for i, p in enumerate(paths, start=1):
            do_log = progress and (not progress_rank0_only or RANK == 0)
            if do_log and (i == 1 or i == total or i % max(1, progress_every) == 0):
                log(f"Opening file {i}/{total}: {p}")
            dsi = xr.open_dataset(p, decode_times=True)
            dsi = _ensure_time_sorted_unique(dsi)
            dsi = _normalize_lon_lat(dsi, lon_system)
            dsi = dsi.sel(time=slice(start, end))
            dsi = _subset_bbox(dsi, bbox, lon_system)
            if "time" in dsi.coords and dsi.time.size == 0:
                continue
            datasets.append(dsi)
        if not datasets:
            raise RuntimeError("No data left after per-file subsetting.")
        ds = xr.concat(datasets, dim="time", data_vars="minimal",
                       coords="minimal", compat="override")
    ds = _ensure_time_sorted_unique(ds)
    ds = _normalize_lon_lat(ds, lon_system)
    ds = ds.sel(time=slice(start, end))
    ds = _subset_bbox(ds, bbox, lon_system)
    return ds


def _grid_score(lon, lat):
    dx = np.median(np.abs(np.diff(lon)))
    dy = np.median(np.abs(np.diff(lat)))
    return float(dx * dy)


def _pick_target_grid(ds_by_key):
    scores = {}
    for key, ds in ds_by_key.items():
        if ds.lon.ndim != 1 or ds.lat.ndim != 1:
            continue
        scores[key] = _grid_score(ds.lon.values, ds.lat.values)
    if not scores:
        raise RuntimeError("Unable to determine target grid (non-1D lon/lat).")
    best = min(scores, key=scores.get)
    return best, ds_by_key[best]


def _interp_if_needed(da, target_lon, target_lat):
    lon = da.coords["lon"].values
    lat = da.coords["lat"].values
    if np.array_equal(lon, target_lon) and np.array_equal(lat, target_lat):
        return da
    return da.interp(lon=target_lon, lat=target_lat, method="linear")


def _stack_slices(times, stack_hours):
    t = pd.to_datetime(times.values)
    if len(t) < 2:
        return [slice(0, len(t))]
    dt_hours = np.median(np.diff(t).astype("timedelta64[s]").astype(float)) / 3600.0
    dt_hours = max(dt_hours, 1e-6)
    per_stack = max(1, int(round(stack_hours / dt_hours)))
    slices = []
    start = 0
    while start < len(t):
        end = min(start + per_stack, len(t))
        slices.append(slice(start, end))
        start = end
    return slices


def _distribute_indices(n, size, rank):
    if size <= 1:
        return list(range(n))
    return [i for i in range(n) if (i % size) == rank]


def _time_base_and_values(times):
    t0 = pd.to_datetime(times.values[0]).to_pydatetime()
    base = datetime(t0.year, t0.month, t0.day)
    tvals = (pd.to_datetime(times.values) - base) / pd.Timedelta("1D")
    units = "days since {} 00:00 UTC".format(base.strftime("%Y-%m-%d"))
    base_date = [base.year, base.month, base.day, 0]
    return base, tvals.astype(float), units, base_date


def _format_index(idx, pad):
    if pad and pad > 0:
        return f"{idx:0{pad}d}"
    return str(idx)


def _write_sflux(kind, idx, times, lon2, lat2, fields, outdir, overwrite=False, index_pad=0):
    idx_txt = _format_index(idx, index_pad)
    fname = os.path.join(outdir, f"sflux_{kind}_1.{idx_txt}.nc")
    if os.path.exists(fname) and not overwrite:
        return fname, False

    base, tvals, units, base_date = _time_base_and_values(times)
    nt = len(tvals)
    ny, nx = lon2.shape

    with Dataset(fname, "w") as nc:
        nc.createDimension("time", nt)
        nc.createDimension("nx_grid", nx)
        nc.createDimension("ny_grid", ny)

        vtime = nc.createVariable("time", "f8", ("time",))
        vtime.long_name = "Time"
        vtime.standard_name = "time"
        vtime.units = units
        vtime.base_date = base_date
        vtime[:] = np.asarray(tvals, dtype="f8")

        vlon = nc.createVariable("lon", "f4", ("ny_grid", "nx_grid"))
        vlon.long_name = "Longitude"
        vlon.standard_name = "longitude"
        vlon.units = "degrees_east"
        vlon[:, :] = lon2.astype("f4")

        vlat = nc.createVariable("lat", "f4", ("ny_grid", "nx_grid"))
        vlat.long_name = "Latitude"
        vlat.standard_name = "latitude"
        vlat.units = "degrees_north"
        vlat[:, :] = lat2.astype("f4")

        if kind == "air":
            meta = dict(
                prmsl=("Pressure reduced to MSL", "air_pressure_at_sea_level", "Pa"),
                spfh=("Surface Specific Humidity (2m AGL)", "specific_humidity", "1"),
                stmp=("Surface Air Temperature (2m AGL)", "air_temperature", "K"),
                uwind=("Surface Eastward Air Velocity (10m AGL)", "eastward_wind", "m/s"),
                vwind=("Surface Northward Air Velocity (10m AGL)", "northward_wind", "m/s"),
            )
        elif kind == "rad":
            meta = dict(
                dlwrf=("Downward Long Wave Radiation Flux", "surface_downwelling_longwave_flux_in_air", "W/m^2"),
                dswrf=("Downward Short Wave Radiation Flux", "surface_downwelling_shortwave_flux_in_air", "W/m^2"),
            )
        elif kind == "prc":
            meta = dict(
                prate=("Surface Precipitation Rate", "precipitation_flux", "kg/m^2/s"),
            )
        else:
            raise ValueError(kind)

        for name, (lname, sname, units_i) in meta.items():
            var = nc.createVariable(name, "f4", ("time", "ny_grid", "nx_grid"))
            var.long_name = lname
            var.standard_name = sname
            var.units = units_i
            var[:, :, :] = np.asarray(fields[name], dtype="f4")

        nc.Conventions = "CF-1.0"

    return fname, True


def _process_window(cfg, start, end, file_index_start):
    # Discover files per key
    files_by_key = {}
    for key, pats in cfg["PATH_GLOBS"].items():
        files = _find_files(cfg["CFS_ROOT"], pats)
        files = _filter_files_by_month(files, start, end)
        files_by_key[key] = files
        if RANK == 0:
            log(f"{key}: {len(files)} files")
    for key, files in files_by_key.items():
        if files:
            continue
        if cfg["MISSING_POLICY"] == "error":
            raise FileNotFoundError(f"No files found for {key}")
        if RANK == 0:
            log(f"Missing files for {key}; policy={cfg['MISSING_POLICY']}")

    # Open datasets per key
    ds_by_key = {}
    for key, paths in files_by_key.items():
        if not paths and cfg["MISSING_POLICY"] != "error":
            continue
        t0 = time.time()
        ds_by_key[key] = _open_dataset(
            paths, cfg["LON_SYSTEM"], start, end, cfg["BBOX"],
            progress=cfg["PROGRESS"], progress_every=cfg["PROGRESS_EVERY"],
            progress_rank0_only=cfg["PROGRESS_RANK0_ONLY"]
        )
        if RANK == 0:
            log(f"Opened {key} in {time.time() - t0:.1f} s")

    if not ds_by_key:
        return file_index_start

    # Align common time
    common_time = None
    for ds in ds_by_key.values():
        t = pd.to_datetime(ds.time.values)
        common_time = t if common_time is None else np.intersect1d(common_time, t)
    if common_time is None or len(common_time) == 0:
        if RANK == 0:
            log("No common times after subsetting; skipping this window.")
        return file_index_start
    for key in ds_by_key:
        ds_by_key[key] = ds_by_key[key].sel(time=common_time)

    # Choose target grid (highest resolution)
    target_key, target_ds = _pick_target_grid(ds_by_key)
    target_lon = target_ds.lon.values
    target_lat = target_ds.lat.values
    if RANK == 0:
        log(f"Target grid: {target_key} ({len(target_lon)} x {len(target_lat)})")

    # Extract fields and interpolate if needed
    def get_da(map_entry):
        key, var = map_entry
        if key not in ds_by_key:
            if cfg["MISSING_POLICY"] == "error":
                raise KeyError(f"Missing dataset for {key}")
            return None
        ds = ds_by_key[key]
        if var not in ds:
            if cfg["MISSING_POLICY"] == "error":
                raise KeyError(f"Variable {var} not in dataset {key}")
            return None
        da = ds[var]
        return _interp_if_needed(da, target_lon, target_lat)

    air = {name: get_da(entry) for name, entry in AIR_MAP.items()}
    rad = {name: get_da(entry) for name, entry in RAD_MAP.items()}
    prc = {name: get_da(entry) for name, entry in PRC_MAP.items()}

    def _fill_missing(group, times, fill_value):
        nt = len(times)
        ny = len(target_lat)
        nx = len(target_lon)
        filled = {}
        for name, da in group.items():
            if da is not None:
                filled[name] = da
                continue
            filled[name] = xr.DataArray(
                np.full((nt, ny, nx), fill_value, dtype=float),
                dims=("time", "lat", "lon"),
                coords=dict(time=times, lat=target_lat, lon=target_lon),
            )
        return filled

    if cfg["MISSING_POLICY"] == "fill":
        air = _fill_missing(air, target_ds.time, cfg["MISSING_FILL"])
        rad = _fill_missing(rad, target_ds.time, cfg["MISSING_FILL"])
        prc = _fill_missing(prc, target_ds.time, cfg["MISSING_FILL"])
    elif cfg["MISSING_POLICY"] == "skip":
        air = {k: v for k, v in air.items() if v is not None}
        rad = {k: v for k, v in rad.items() if v is not None}
        prc = {k: v for k, v in prc.items() if v is not None}

    # Apply output lon system (WGS84 default: -180..180) and align lon ordering.
    target_ds, grouped = _apply_output_lon_order(
        target_ds, {"air": air, "rad": rad, "prc": prc}, cfg["OUTPUT_LON_SYSTEM"]
    )
    air = grouped["air"]
    rad = grouped["rad"]
    prc = grouped["prc"]

    target_lon = target_ds.lon.values
    target_lat = target_ds.lat.values
    lon2, lat2 = np.meshgrid(target_lon, target_lat)

    # Stack slices + MPI distribution
    slices = _stack_slices(target_ds.time, cfg["STACK_HOURS"])
    my_idxs = _distribute_indices(len(slices), SIZE if cfg["USE_MPI"] else 1, RANK)
    log(f"Assigned {len(my_idxs)} stack(s) of {len(slices)} total.")

    for i in my_idxs:
        slc = slices[i]
        times = target_ds.time.isel(time=slc)
        air_fields = {k: air[k].isel(time=slc).values for k in air}
        rad_fields = {k: rad[k].isel(time=slc).values for k in rad}
        prc_fields = {k: prc[k].isel(time=slc).values for k in prc}

        file_index = file_index_start + i
        f1, _ = _write_sflux(
            "air", file_index, times, lon2, lat2, air_fields, cfg["OUTDIR"],
            cfg["OVERWRITE"], index_pad=cfg["INDEX_PAD"]
        )
        f2, _ = _write_sflux(
            "rad", file_index, times, lon2, lat2, rad_fields, cfg["OUTDIR"],
            cfg["OVERWRITE"], index_pad=cfg["INDEX_PAD"]
        )
        f3, _ = _write_sflux(
            "prc", file_index, times, lon2, lat2, prc_fields, cfg["OUTDIR"],
            cfg["OVERWRITE"], index_pad=cfg["INDEX_PAD"]
        )
        log(f"Wrote stack {file_index:04d}: {f1}, {f2}, {f3}")

    return file_index_start + len(slices)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Create SCHISM sflux from CFSv2 (subset + interpolate-if-needed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cfs-root", help="Root directory containing yearly CFSv2 folders.")
    ap.add_argument("--start", help="Start datetime (YYYY-MM-DD HH:MM:SS).")
    ap.add_argument("--end", help="End datetime (YYYY-MM-DD HH:MM:SS).")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    ap.add_argument("--outdir", help="Output directory for sflux files.")
    ap.add_argument("--stack-hours", type=int, help="Hours per stack file.")
    ap.add_argument("--lon-system", choices=("360", "180"), help="Longitude system to use.")
    ap.add_argument("--index-pad", type=int,
                    help="Zero padding width for file index (0 -> no padding).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--use-mpi", action="store_true", help="Enable MPI if available.")
    ap.add_argument("--stream-monthly", action="store_true",
                    help="Process and write outputs month-by-month.")
    ap.add_argument("--output-lon-system", choices=("360", "180"),
                    help="Output lon system for sflux files.")
    ap.add_argument("--missing-policy", choices=("error", "fill", "skip"),
                    help="How to handle missing variables/files.")
    ap.add_argument("--missing-fill", type=float,
                    help="Fill value when missing-policy=fill (default: NaN).")
    ap.add_argument("--progress", action="store_true",
                    help="Print progress while opening files (fallback mode).")
    ap.add_argument("--progress-every", type=int,
                    help="Print every N files when --progress is set.")
    ap.add_argument("--progress-all", action="store_true",
                    help="Print progress from all ranks (default: rank 0 only).")
    ap.add_argument("--config", help="Optional JSON config file to override defaults.")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cfg = dict(CONFIG)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    if args.cfs_root:
        cfg["CFS_ROOT"] = args.cfs_root
    if args.start:
        cfg["START"] = args.start
    if args.end:
        cfg["END"] = args.end
    if args.bbox:
        cfg["BBOX"] = dict(lon_min=args.bbox[0], lon_max=args.bbox[1],
                           lat_min=args.bbox[2], lat_max=args.bbox[3])
    if args.outdir:
        cfg["OUTDIR"] = args.outdir
    if args.stack_hours:
        cfg["STACK_HOURS"] = int(args.stack_hours)
    if args.lon_system:
        cfg["LON_SYSTEM"] = args.lon_system
    if args.output_lon_system:
        cfg["OUTPUT_LON_SYSTEM"] = args.output_lon_system
    if args.index_pad is not None:
        cfg["INDEX_PAD"] = int(args.index_pad)
    if args.overwrite:
        cfg["OVERWRITE"] = True
    if args.use_mpi:
        cfg["USE_MPI"] = True
    if args.stream_monthly:
        cfg["STREAM_MODE"] = "month"
    if args.missing_policy:
        cfg["MISSING_POLICY"] = args.missing_policy
    if args.missing_fill is not None:
        cfg["MISSING_FILL"] = float(args.missing_fill)
    if args.progress:
        cfg["PROGRESS"] = True
    if args.progress_every is not None:
        cfg["PROGRESS_EVERY"] = int(args.progress_every)
    if args.progress_all:
        cfg["PROGRESS_RANK0_ONLY"] = False

    if SIZE > 1 and not cfg["USE_MPI"]:
        log("MPI detected but disabled by config; running with rank 0 only.")

    ensure_dir(cfg["OUTDIR"])
    if RANK == 0:
        with open(os.path.join(cfg["OUTDIR"], "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    if cfg["STREAM_MODE"] == "month":
        file_index = 1
        for yyyymm in sorted(_month_range(cfg["START"], cfg["END"])):
            if RANK == 0:
                log(f"Streaming month {yyyymm}")
            start, end = _month_start_end(yyyymm)
            file_index = _process_window(cfg, start, end, file_index)
    else:
        _process_window(cfg, cfg["START"], cfg["END"], 1)

    if cfg["USE_MPI"] and COMM is not None:
        COMM.barrier()
    if RANK == 0:
        log("Done.")


if __name__ == "__main__":
    main()
