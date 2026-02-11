#!/usr/bin/env python3
"""
Create SCHISM sflux radiation files from ERA5 accumulated forecast fields.

This script writes:
  - sflux_rad_1.N.nc (variables: dlwrf, dswrf)

Input expectations:
  - ERA5 ssrd/strd netCDF files under ERA5_ROOT (flat or nested directories)
  - File discovery patterns:
      **/*ssrd*.nc
      **/*strd*.nc
  - Dataset style (RDA ds633.0):
      dims: forecast_initial_time, forecast_hour, latitude, longitude
      vars: SSRD/ssrd, STRD/strd
      units: W m**-2 s (accumulated energy)

Radiation conversion:
  - hourly_flux(hour1) = accum(hour1) / 3600
  - hourly_flux(hourn) = (accum(hourn) - accum(hour[n-1])) / 3600
  - Output units: W/m^2

Streaming mode:
  - JMA-like stack streaming. Each stack window (e.g., one day when
    STACK_HOURS=24) is processed independently and written immediately when
    ready.
  - By default, incomplete stacks are skipped and can be produced on a later
    rerun when more ERA5 files arrive.
  - By default, converted data is released after each stack (memory-first).
    Use --reuse-cache only when speed is preferred over memory.

Flags quick guide:
  --era5-root
    Root folder for ERA5 ssrd/strd files.
    Example: --era5-root /S/data00/G6008/d1041/dataset/ERA5

  --start / --end
    Output period (inclusive), format: "YYYY-MM-DD HH:MM:SS".
    Example: --start "2017-01-01 00:00:00" --end "2017-12-31 23:00:00"

  --bbox LON_MIN LON_MAX LAT_MIN LAT_MAX
    Spatial subset before writing output.
    Example: --bbox 140 142 37 39

  --outdir
    Output folder for sflux_rad files.
    Example: --outdir ./sflux_era5_sendai

  --stack-hours
    Hours per output file. 24 => daily output, 72 => 3-day output.
    Example: --stack-hours 24

  --index-pad
    Zero-padding width for file index.
    Example: --index-pad 4 -> sflux_rad_1.0001.nc

  --lon-system
    Input longitude convention of ERA5 files.
      360: [0, 360)
      180: [-180, 180)

  --output-lon-system
    Output longitude convention for SCHISM sflux.
      180 or 360

  --stream-mode
    month: process month windows (default; memory safer)
    all: process whole period as one window

  --allow-partial-stack
    If set, write stack even if not all expected hourly timesteps exist.
    Default behavior is strict (skip incomplete stacks).

  --reuse-cache
    Reuse converted fields across adjacent stacks with identical input files.
    Faster, but higher memory use.

  --mpi-distribution
    How stacks are distributed across ranks when --use-mpi is enabled:
      key    : group adjacent stacks by input-file key, then balance groups
      block  : contiguous stack blocks per rank
      cyclic : round-robin (i % nproc)

  --overwrite
    Overwrite existing sflux_rad_1.N.nc files.

  --progress / --progress-every
    Print per-file opening progress.
    Example: --progress --progress-every 5

Examples:
  1) Standard daily generation (strict readiness):
     python make_sflux_era5.py \
       --era5-root ./ERA5 \
       --start "2017-01-01 00:00:00" \
       --end   "2017-01-31 23:00:00" \
       --bbox 140 142 37 39 \
       --outdir ./sflux_era5_jan \
       --stack-hours 24 \
       --stream-mode month

  2) Allow partial stacks during early testing:
     python make_sflux_era5.py \
       --era5-root ./ERA5 \
       --start "2017-01-01 00:00:00" \
       --end   "2017-01-31 23:00:00" \
       --outdir ./sflux_era5_partial \
       --stack-hours 24 \
       --allow-partial-stack
"""

import argparse
import gc
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
    from mpi4py import MPI  # type: ignore
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1


CONFIG = dict(
    ERA5_ROOT="../ERA5",
    START="2012-01-01 00:00:00",
    END="2014-12-31 23:00:00",
    BBOX=dict(lon_min=140.0, lon_max=142.0, lat_min=37.0, lat_max=39.0),
    OUTDIR="./sflux_out",
    STACK_HOURS=24,
    INDEX_PAD=0,  # 0 -> sflux_rad_1.3.nc ; 4 -> sflux_rad_1.0003.nc
    LON_SYSTEM="360",  # input lon system ("360" for ERA5 0..360)
    OUTPUT_LON_SYSTEM="180",  # output lon system expected by SCHISM
    OVERWRITE=False,
    STREAM_MODE="month",  # "month" or "all"
    USE_MPI=False,
    MPI_DISTRIBUTION="key",  # key, block, cyclic
    PROGRESS=True,
    PROGRESS_EVERY=1,
    REQUIRE_FULL_STACK=True,  # if True, skip stack unless all expected hours exist
    REUSE_CACHE=False,  # if True, reuse converted fields across adjacent stacks
    PATH_GLOBS=dict(
        ssrd=["**/*ssrd.*.nc", "**/*SSRD*.nc"],
        strd=["**/*strd.*.nc", "**/*STRD*.nc"],
    ),
)


def log(msg, rank0_only=False):
    if rank0_only and RANK != 0:
        return
    prefix = f"[rank {RANK}/{SIZE}] " if SIZE > 1 else ""
    print(prefix + str(msg), flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _find_files(root, patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    return sorted(set(files))


def _parse_file_span(fp):
    # Example: ...2013040106_2013041606.nc
    m = re.search(r"\.(\d{10})_(\d{10})\.nc$", os.path.basename(fp))
    if m is None:
        return None
    t0 = pd.to_datetime(m.group(1), format="%Y%m%d%H")
    t1 = pd.to_datetime(m.group(2), format="%Y%m%d%H")
    return t0, t1


def _filter_files_by_window(files, start, end, pad_hours=24):
    s = pd.to_datetime(start) - pd.Timedelta(hours=pad_hours)
    e = pd.to_datetime(end) + pd.Timedelta(hours=pad_hours)
    out = []
    for fp in files:
        span = _parse_file_span(fp)
        if span is None:
            out.append(fp)
            continue
        t0, t1 = span
        if t1 >= s and t0 <= e:
            out.append(fp)
    return out


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
    return out


def _month_start_end(yyyymm):
    s = pd.Timestamp(f"{yyyymm}01 00:00:00")
    e = (s + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
    return s, e


def _normalize_lon(lon, system):
    lon = np.asarray(lon, dtype=float)
    if system == "360":
        return lon % 360.0
    if system == "180":
        return ((lon + 180.0) % 360.0) - 180.0
    raise ValueError("Longitude system must be '360' or '180'.")


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


def _rename_era5_coords(ds):
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    if "forecast_initial_time" not in ds.coords:
        for cand in ("forecast_reference_time", "time"):
            if cand in ds.coords:
                ds = ds.rename({cand: "forecast_initial_time"})
                break
    if "forecast_hour" not in ds.coords:
        for cand in ("step", "leadtime"):
            if cand in ds.coords:
                ds = ds.rename({cand: "forecast_hour"})
                break
    return ds


def _normalize_lon_lat(ds, lon_system):
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("Dataset missing lon/lat coordinates.")
    ds = ds.assign_coords(lon=_normalize_lon(ds.lon.values, lon_system))
    ds = ds.sortby("lon")
    if not np.all(np.diff(ds.lat.values) > 0):
        ds = ds.sortby("lat")
    return ds


def _ensure_fitime_sorted_unique(ds):
    if "forecast_initial_time" not in ds.coords:
        return ds
    t = pd.to_datetime(ds["forecast_initial_time"].values)
    order = np.argsort(t.values)
    t_sorted = t.values[order]
    keep = ~pd.Index(t_sorted).duplicated(keep="last")
    ds = ds.isel(forecast_initial_time=order)
    ds = ds.isel(forecast_initial_time=keep)
    ds = ds.assign_coords(forecast_initial_time=t_sorted[keep])
    return ds


def _pick_var_name(ds, candidates):
    for name in candidates:
        if name in ds.data_vars:
            return name
    lower_map = {k.lower(): k for k in ds.data_vars}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(f"None of candidates {candidates} found in {list(ds.data_vars)}")


def _open_era5(paths, lon_system, bbox, progress=False, progress_every=10):
    if not paths:
        raise FileNotFoundError("No ERA5 input files were provided.")
    dsets = []
    total = len(paths)
    for i, fp in enumerate(paths, start=1):
        if progress and (i == 1 or i == total or i % max(1, progress_every) == 0):
            log(f"Opening file {i}/{total}: {fp}")
        dsi = xr.open_dataset(fp, decode_times=True)
        dsi = _rename_era5_coords(dsi)
        dsi = _normalize_lon_lat(dsi, lon_system)
        dsi = _subset_bbox(dsi, bbox, lon_system)
        dsets.append(dsi)
    ds = xr.concat(
        dsets,
        dim="forecast_initial_time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    return _ensure_fitime_sorted_unique(ds)


def _accum_to_hourly_flux(ds, var_name):
    if "forecast_initial_time" not in ds.coords or "forecast_hour" not in ds.coords:
        raise ValueError("Dataset must have forecast_initial_time and forecast_hour.")

    da = ds[var_name].transpose("forecast_initial_time", "forecast_hour", "lat", "lon")
    first = da.isel(forecast_hour=0)
    diff = da.diff("forecast_hour")
    incr = xr.concat([first, diff], dim="forecast_hour")
    incr = incr.assign_coords(forecast_hour=da["forecast_hour"])

    # ERA5 accumulated radiation is in J/m^2 (W m^-2 s); convert to W/m^2.
    flux = incr / 3600.0
    flux = flux.where(flux >= 0.0, 0.0)

    fit = pd.to_datetime(ds["forecast_initial_time"].values).values.astype("datetime64[h]")
    fhr = np.asarray(ds["forecast_hour"].values, dtype=int)
    valid_2d = fit[:, None] + fhr[None, :].astype("timedelta64[h]")

    flat = xr.DataArray(
        flux.values.reshape(-1, flux.sizes["lat"], flux.sizes["lon"]),
        dims=("time", "lat", "lon"),
        coords=dict(
            time=valid_2d.reshape(-1).astype("datetime64[ns]"),
            lat=flux["lat"].values,
            lon=flux["lon"].values,
        ),
        name=var_name,
    )

    t = pd.to_datetime(flat["time"].values)
    order = np.argsort(t.values)
    t_sorted = t.values[order]
    # Keep last duplicate to prefer newer files/cycles for overlapping valid times.
    keep = ~pd.Index(t_sorted).duplicated(keep="last")
    flat = flat.isel(time=order).isel(time=keep)
    flat = flat.assign_coords(time=t_sorted[keep])
    return flat.sortby("time")


def _apply_output_lon(da, output_system):
    if output_system is None:
        return da
    lon = _normalize_lon(da.lon.values, output_system)
    return da.assign_coords(lon=lon).sortby("lon")


def _stack_slices(times, stack_hours):
    t = pd.to_datetime(times.values)
    if len(t) < 2:
        return [slice(0, len(t))]
    dt_hours = np.median(np.diff(t).astype("timedelta64[s]").astype(float)) / 3600.0
    dt_hours = max(dt_hours, 1e-6)
    per_stack = max(1, int(round(stack_hours / dt_hours)))
    out = []
    start = 0
    while start < len(t):
        end = min(start + per_stack, len(t))
        out.append(slice(start, end))
        start = end
    return out


def _make_hourly_schedule(start, end):
    return pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="1h")


def _time_base_and_values(times):
    t0 = pd.to_datetime(times.values[0]).to_pydatetime()
    base = datetime(t0.year, t0.month, t0.day)
    tvals = (pd.to_datetime(times.values) - base) / pd.Timedelta("1D")
    units = f"days since {base.strftime('%Y-%m-%d')}"
    base_date = [base.year, base.month, base.day, 0]
    return tvals.astype(float), units, base_date


def _format_index(idx, pad):
    if pad and pad > 0:
        return f"{idx:0{pad}d}"
    return str(idx)


def _write_sflux_rad(idx, times, lon2, lat2, dlwrf, dswrf, outdir, overwrite=False, index_pad=0):
    idx_txt = _format_index(idx, index_pad)
    fname = os.path.join(outdir, f"sflux_rad_1.{idx_txt}.nc")
    if os.path.exists(fname) and not overwrite:
        return fname, False

    tvals, units, base_date = _time_base_and_values(times)
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

        vlon = nc.createVariable("lon", "f8", ("ny_grid", "nx_grid"))
        vlon.long_name = "Longitude"
        vlon.standard_name = "longitude"
        vlon.units = "degrees_east"
        vlon[:, :] = np.asarray(lon2, dtype="f8")

        vlat = nc.createVariable("lat", "f8", ("ny_grid", "nx_grid"))
        vlat.long_name = "Latitude"
        vlat.standard_name = "latitude"
        vlat.units = "degrees_north"
        vlat[:, :] = np.asarray(lat2, dtype="f8")

        vdl = nc.createVariable("dlwrf", "f4", ("time", "ny_grid", "nx_grid"))
        vdl.long_name = "Downward Long Wave Radiation Flux"
        vdl.standard_name = "surface_downwelling_longwave_flux_in_air"
        vdl.units = "W/m^2"
        vdl[:, :, :] = np.asarray(dlwrf, dtype="f4")

        vds = nc.createVariable("dswrf", "f4", ("time", "ny_grid", "nx_grid"))
        vds.long_name = "Downward Short Wave Radiation Flux"
        vds.standard_name = "surface_downwelling_shortwave_flux_in_air"
        vds.units = "W/m^2"
        vds[:, :, :] = np.asarray(dswrf, dtype="f4")

        # netCDF4 reserves Dataset.file_format; do not assign it directly.
        # Keep optional metadata via global attrs when supported.
        try:
            nc.setncattr("file_format", "NETCDF4")
        except Exception:
            pass
        nc.dimname = ["nx_grid", "ny_grid", "time"]

    return fname, True


def _build_rad_cache_entry(cfg, ssrd_files, strd_files, materialize=False):
    t0 = time.time()
    ds_ssrd = _open_era5(
        ssrd_files,
        lon_system=cfg["LON_SYSTEM"],
        bbox=cfg["BBOX"],
        progress=cfg["PROGRESS"],
        progress_every=cfg["PROGRESS_EVERY"],
    )
    ds_strd = _open_era5(
        strd_files,
        lon_system=cfg["LON_SYSTEM"],
        bbox=cfg["BBOX"],
        progress=cfg["PROGRESS"],
        progress_every=cfg["PROGRESS_EVERY"],
    )
    log(f"    Opened datasets in {time.time() - t0:.1f} s")

    try:
        ssrd_name = _pick_var_name(ds_ssrd, ["SSRD", "ssrd"])
        strd_name = _pick_var_name(ds_strd, ["STRD", "strd"])

        t1 = time.time()
        da_dswrf = _accum_to_hourly_flux(ds_ssrd, ssrd_name)
        da_dlwrf = _accum_to_hourly_flux(ds_strd, strd_name)
        log(f"    Converted accumulated fields in {time.time() - t1:.1f} s")

        common_time = np.intersect1d(
            pd.to_datetime(da_dswrf.time.values).values,
            pd.to_datetime(da_dlwrf.time.values).values,
        )
        if len(common_time) == 0:
            raise RuntimeError("No common times between ssrd and strd after conversion.")

        da_dswrf = da_dswrf.sel(time=common_time)
        da_dlwrf = da_dlwrf.sel(time=common_time)

        da_dswrf = _apply_output_lon(da_dswrf, cfg["OUTPUT_LON_SYSTEM"])
        da_dlwrf = _apply_output_lon(da_dlwrf, cfg["OUTPUT_LON_SYSTEM"])
        da_dlwrf = da_dlwrf.sel(lon=da_dswrf.lon, lat=da_dswrf.lat)
        if materialize:
            da_dswrf = da_dswrf.load()
            da_dlwrf = da_dlwrf.load()

        lon = da_dswrf.lon.values
        lat = da_dswrf.lat.values
        lon2, lat2 = np.meshgrid(lon, lat)
        return dict(dswrf=da_dswrf, dlwrf=da_dlwrf, lon2=lon2, lat2=lat2)
    finally:
        ds_ssrd.close()
        ds_strd.close()


def _extract_stack_from_cache(entry, start, end, require_full_stack=True):
    da_dswrf = entry["dswrf"].sel(time=slice(pd.to_datetime(start), pd.to_datetime(end)))
    da_dlwrf = entry["dlwrf"].sel(time=slice(pd.to_datetime(start), pd.to_datetime(end)))
    if da_dswrf.time.size == 0:
        raise RuntimeError("No time samples remain after filtering.")

    expected = pd.to_datetime(_make_hourly_schedule(start, end))
    if bool(require_full_stack):
        missing = expected.difference(pd.to_datetime(da_dswrf.time.values))
        if len(missing) > 0:
            raise RuntimeError(
                f"Missing {len(missing)} hourly sample(s); first missing={missing[0]}"
            )
    da_dswrf = da_dswrf.reindex(time=expected)
    da_dlwrf = da_dlwrf.reindex(time=expected)
    return expected, entry["lon2"], entry["lat2"], da_dlwrf.values, da_dswrf.values


def _release_cache_state(cache_state):
    if cache_state is None:
        return
    cache_state["key"] = None
    cache_state["entry"] = None
    gc.collect()


def _assign_block_indices(n, rank, size):
    base = n // size
    rem = n % size
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    return set(range(start, start + count))


def _assign_key_groups(keys, size):
    # Group contiguous stacks with the same file-key, then assign groups
    # greedily to the least-loaded rank.
    if not keys:
        return [set() for _ in range(size)]
    groups = []
    g_start = 0
    g_key = keys[0]
    for i in range(1, len(keys)):
        if keys[i] != g_key:
            groups.append(list(range(g_start, i)))
            g_start = i
            g_key = keys[i]
    groups.append(list(range(g_start, len(keys))))

    assigns = [set() for _ in range(size)]
    loads = [0] * size
    for g in groups:
        r = min(range(size), key=lambda x: loads[x])
        assigns[r].update(g)
        loads[r] += len(g)
    return assigns


def _process_single_stack(cfg, idx, s0, s1, ssrd_all, strd_all, cache_state=None):
    ssrd_files = _filter_files_by_window(ssrd_all, s0, s1, pad_hours=24)
    strd_files = _filter_files_by_window(strd_all, s0, s1, pad_hours=24)
    log(f"  Stack {idx}: {s0} -> {s1}")
    log(f"    ssrd files: {len(ssrd_files)}")
    log(f"    strd files: {len(strd_files)}")
    if not ssrd_files or not strd_files:
        log("    [skip] missing ssrd/strd files.")
        return

    key = (tuple(ssrd_files), tuple(strd_files))
    use_cache = bool(cfg.get("REUSE_CACHE", False)) and cache_state is not None
    try:
        if use_cache and cache_state.get("key") == key:
            entry = cache_state["entry"]
            log("    Using cached converted fields.")
        else:
            if use_cache and cache_state.get("key") is not None and cache_state.get("key") != key:
                _release_cache_state(cache_state)
            entry = _build_rad_cache_entry(
                cfg,
                ssrd_files,
                strd_files,
                materialize=use_cache,
            )
            if use_cache:
                cache_state["key"] = key
                cache_state["entry"] = entry

        times, lon2, lat2, dlwrf, dswrf = _extract_stack_from_cache(
            entry, s0, s1, require_full_stack=cfg.get("REQUIRE_FULL_STACK", True)
        )
    except Exception as exc:
        log(f"    [skip] not ready: {exc}")
        return

    fname, wrote = _write_sflux_rad(
        idx=idx,
        times=times,
        lon2=lon2,
        lat2=lat2,
        dlwrf=dlwrf,
        dswrf=dswrf,
        outdir=cfg["OUTDIR"],
        overwrite=cfg["OVERWRITE"],
        index_pad=cfg["INDEX_PAD"],
    )
    status = "wrote" if wrote else "exists"
    log(f"    [{status}] {fname}")

    if not use_cache:
        del entry, times, lon2, lat2, dlwrf, dswrf
        gc.collect()


def _process_window(cfg, start, end, file_index_start, ssrd_all, strd_all,
                    stack_rank=0, stack_size=1):
    log(f"Window {start} -> {end}", rank0_only=(stack_size > 1))
    out_times = _make_hourly_schedule(start, end)
    slices = _stack_slices(out_times, cfg["STACK_HOURS"])
    if not slices:
        log("  No stack slices in this window; skipping.", rank0_only=(stack_size > 1))
        return file_index_start

    stack_meta = []
    for i, slc in enumerate(slices):
        tt = out_times[slc]
        s0 = pd.to_datetime(tt[0]).strftime("%Y-%m-%d %H:%M:%S")
        s1 = pd.to_datetime(tt[-1]).strftime("%Y-%m-%d %H:%M:%S")
        stack_meta.append((i, s0, s1))

    if stack_size <= 1:
        my_indices = set(range(len(slices)))
        dist_mode = "serial"
    else:
        dist_mode = str(cfg.get("MPI_DISTRIBUTION", "key")).lower().strip()
        if dist_mode == "cyclic":
            my_indices = {i for i in range(len(slices)) if (i % stack_size) == stack_rank}
        elif dist_mode == "block":
            my_indices = _assign_block_indices(len(slices), stack_rank, stack_size)
        else:
            # cache-aware distribution (default for MPI): keep same-file stacks together
            keys = []
            for _, s0, s1 in stack_meta:
                ss = tuple(_filter_files_by_window(ssrd_all, s0, s1, pad_hours=24))
                st = tuple(_filter_files_by_window(strd_all, s0, s1, pad_hours=24))
                keys.append((ss, st))
            assigns = _assign_key_groups(keys, stack_size)
            my_indices = assigns[stack_rank]
            dist_mode = "key"

    my_count = len(my_indices)
    log(
        f"  Processing {len(slices)} stack(s), assigned {my_count} to this rank "
        f"(distribution={dist_mode})",
        rank0_only=(stack_size <= 1),
    )
    cache_state = dict(key=None, entry=None) if cfg.get("REUSE_CACHE", False) else None
    for i, s0, s1 in stack_meta:
        if i not in my_indices:
            continue
        idx = file_index_start + i
        _process_single_stack(
            cfg=cfg,
            idx=idx,
            s0=s0,
            s1=s1,
            ssrd_all=ssrd_all,
            strd_all=strd_all,
            cache_state=cache_state,
        )

    _release_cache_state(cache_state)
    return file_index_start + len(slices)


def _parse_args(argv=None):
    class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    epilog = (
        "Examples:\n"
        "  python make_sflux_era5.py \\\n"
        "    --era5-root ./ERA5 \\\n"
        "    --start '2017-01-01 00:00:00' --end '2017-01-31 23:00:00' \\\n"
        "    --bbox 140 142 37 39 --outdir ./sflux_era5_jan \\\n"
        "    --stack-hours 24 --stream-mode month\n\n"
        "  python make_sflux_era5.py \\\n"
        "    --era5-root ./ERA5 --start '2017-01-01 00:00:00' \\\n"
        "    --end '2017-01-31 23:00:00' --outdir ./sflux_era5_partial \\\n"
        "    --allow-partial-stack --overwrite\n\n"
        "  python make_sflux_era5.py \\\n"
        "    --era5-root ./ERA5 --start '2017-01-01 00:00:00' \\\n"
        "    --end '2017-01-31 23:00:00' --outdir ./sflux_era5_fast \\\n"
        "    --reuse-cache\n\n"
        "  mpirun -np 8 python make_sflux_era5.py \\\n"
        "    --use-mpi --era5-root ./ERA5 \\\n"
        "    --start '2017-01-01 00:00:00' --end '2017-12-31 23:00:00' \\\n"
        "    --outdir ./sflux_era5_2017 --stack-hours 24 \\\n"
        "    --mpi-distribution key\n\n"
        "  python make_sflux_era5.py --config config_era5_rad.json"
    )

    ap = argparse.ArgumentParser(
        description="Create SCHISM sflux_rad files from ERA5 ssrd/strd.",
        formatter_class=_Formatter,
        epilog=epilog,
    )
    ap.add_argument(
        "--era5-root",
        help=(
            "Root directory containing ERA5 ssrd/strd files. "
            "File discovery is recursive."
        ),
    )
    ap.add_argument(
        "--start",
        help="Start datetime (inclusive), format: YYYY-MM-DD HH:MM:SS.",
    )
    ap.add_argument(
        "--end",
        help="End datetime (inclusive), format: YYYY-MM-DD HH:MM:SS.",
    )
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help=(
            "Spatial subset before writing output. "
            "Use same lon convention as --lon-system."
        ),
    )
    ap.add_argument(
        "--outdir",
        help="Output directory for sflux_rad_1.N.nc files and config snapshot.",
    )
    ap.add_argument(
        "--stack-hours",
        type=int,
        help="Hours per output stack file (24=daily, 72=3-day, ...).",
    )
    ap.add_argument(
        "--index-pad",
        type=int,
        help="Zero-padding width for output file index.",
    )
    ap.add_argument(
        "--lon-system",
        choices=("360", "180"),
        help="Input ERA5 longitude system.",
    )
    ap.add_argument(
        "--output-lon-system",
        choices=("360", "180"),
        help="Output longitude system for SCHISM sflux files.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files if present.",
    )
    ap.add_argument(
        "--use-mpi",
        action="store_true",
        help=(
            "Enable MPI parallel execution (requires mpi4py + mpirun/srun).\n"
            "Stacks are distributed across ranks."
        ),
    )
    ap.add_argument(
        "--mpi-distribution",
        choices=("key", "block", "cyclic"),
        help=(
            "Stack distribution strategy when MPI is enabled.\n"
            "  key   : cache-aware grouping by input-file key (recommended)\n"
            "  block : contiguous stack blocks per rank\n"
            "  cyclic: round-robin by stack index"
        ),
    )
    ap.add_argument(
        "--stream-mode",
        choices=("month", "all"),
        help=(
            "Streaming mode.\n"
            "  month: process by month windows (memory safer)\n"
            "  all  : process full START..END in one window"
        ),
    )
    ap.add_argument(
        "--stream-monthly",
        action="store_true",
        help="Deprecated alias for --stream-mode month.",
    )
    ap.add_argument(
        "--allow-partial-stack",
        action="store_true",
        help=(
            "Allow writing stack even with missing hours.\n"
            "Default behavior skips incomplete stacks until data is complete."
        ),
    )
    ap.add_argument(
        "--reuse-cache",
        action="store_true",
        help=(
            "Reuse converted fields across adjacent stacks when source files are identical.\n"
            "Faster, but uses more memory."
        ),
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help=(
            "Print file-opening progress while scanning/loading ERA5 files.\n"
            "Useful for long runs."
        ),
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        help="When --progress is enabled, print every N files.",
    )
    ap.add_argument(
        "--config",
        help=(
            "Optional JSON config file for bulk overrides. "
            "CLI flags still override config values."
        ),
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    cfg = dict(CONFIG)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    if args.era5_root:
        cfg["ERA5_ROOT"] = args.era5_root
    if args.start:
        cfg["START"] = args.start
    if args.end:
        cfg["END"] = args.end
    if args.bbox:
        cfg["BBOX"] = dict(
            lon_min=args.bbox[0],
            lon_max=args.bbox[1],
            lat_min=args.bbox[2],
            lat_max=args.bbox[3],
        )
    if args.outdir:
        cfg["OUTDIR"] = args.outdir
    if args.stack_hours:
        cfg["STACK_HOURS"] = int(args.stack_hours)
    if args.index_pad is not None:
        cfg["INDEX_PAD"] = int(args.index_pad)
    if args.lon_system:
        cfg["LON_SYSTEM"] = args.lon_system
    if args.output_lon_system:
        cfg["OUTPUT_LON_SYSTEM"] = args.output_lon_system
    if args.overwrite:
        cfg["OVERWRITE"] = True
    if args.use_mpi:
        cfg["USE_MPI"] = True
    if args.mpi_distribution:
        cfg["MPI_DISTRIBUTION"] = args.mpi_distribution
    if args.stream_mode:
        cfg["STREAM_MODE"] = args.stream_mode
    if args.stream_monthly:
        cfg["STREAM_MODE"] = "month"
    if args.allow_partial_stack:
        cfg["REQUIRE_FULL_STACK"] = False
    if args.reuse_cache:
        cfg["REUSE_CACHE"] = True
    if args.progress:
        cfg["PROGRESS"] = True
    if args.progress_every is not None:
        cfg["PROGRESS_EVERY"] = int(args.progress_every)

    if SIZE > 1 and not cfg["USE_MPI"]:
        log("MPI ranks detected; enabling MPI mode automatically.", rank0_only=True)
        cfg["USE_MPI"] = True
    use_mpi = bool(cfg["USE_MPI"]) and SIZE > 1

    if RANK == 0:
        ensure_dir(cfg["OUTDIR"])
        with open(os.path.join(cfg["OUTDIR"], "config_used_era5.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    if use_mpi and COMM is not None:
        COMM.Barrier()

    if use_mpi and COMM is not None:
        if RANK == 0:
            ssrd_all = _find_files(cfg["ERA5_ROOT"], cfg["PATH_GLOBS"]["ssrd"])
            strd_all = _find_files(cfg["ERA5_ROOT"], cfg["PATH_GLOBS"]["strd"])
            err_msg = None
            if not ssrd_all or not strd_all:
                err_msg = "No ssrd/strd files found under ERA5_ROOT."
            log(f"Discovered ssrd files: {len(ssrd_all)}", rank0_only=True)
            log(f"Discovered strd files: {len(strd_all)}", rank0_only=True)
        else:
            ssrd_all = None
            strd_all = None
            err_msg = None
        ssrd_all = COMM.bcast(ssrd_all, root=0)
        strd_all = COMM.bcast(strd_all, root=0)
        err_msg = COMM.bcast(err_msg, root=0)
        if err_msg:
            raise FileNotFoundError(err_msg)
    else:
        ssrd_all = _find_files(cfg["ERA5_ROOT"], cfg["PATH_GLOBS"]["ssrd"])
        strd_all = _find_files(cfg["ERA5_ROOT"], cfg["PATH_GLOBS"]["strd"])
        log(f"Discovered ssrd files: {len(ssrd_all)}")
        log(f"Discovered strd files: {len(strd_all)}")
        if not ssrd_all or not strd_all:
            raise FileNotFoundError("No ssrd/strd files found under ERA5_ROOT.")

    start = pd.to_datetime(cfg["START"])
    end = pd.to_datetime(cfg["END"])
    if end < start:
        raise ValueError("END must be >= START.")

    file_index = 1
    if cfg["STREAM_MODE"] == "month":
        for yyyymm in _month_range(start, end):
            mstart, mend = _month_start_end(yyyymm)
            wstart = max(start, mstart)
            wend = min(end, mend)
            if wend < wstart:
                continue
            file_index = _process_window(
                cfg=cfg,
                start=wstart.strftime("%Y-%m-%d %H:%M:%S"),
                end=wend.strftime("%Y-%m-%d %H:%M:%S"),
                file_index_start=file_index,
                ssrd_all=ssrd_all,
                strd_all=strd_all,
                stack_rank=(RANK if use_mpi else 0),
                stack_size=(SIZE if use_mpi else 1),
            )
    else:
        _process_window(
            cfg=cfg,
            start=start.strftime("%Y-%m-%d %H:%M:%S"),
            end=end.strftime("%Y-%m-%d %H:%M:%S"),
            file_index_start=file_index,
            ssrd_all=ssrd_all,
            strd_all=strd_all,
            stack_rank=(RANK if use_mpi else 0),
            stack_size=(SIZE if use_mpi else 1),
        )

    if use_mpi and COMM is not None:
        COMM.Barrier()
    log("Done.", rank0_only=True)


if __name__ == "__main__":
    main()
