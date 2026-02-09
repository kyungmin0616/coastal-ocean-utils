#!/usr/bin/env python3
"""
Create SCHISM sflux files from JMA MSM GRIB2 files.

This script generates:
  - sflux_air_1.NNNN.nc (prmsl, spfh, stmp, uwind, vwind)
  - sflux_prc_1.NNNN.nc (prate)

It intentionally does NOT generate:
  - sflux_rad_1.NNNN.nc

Input layout example:
  msm_data/
    2011/
      Z__C_RJTD_20111231000000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin
      ...

Notes:
  - Relative humidity is converted to specific humidity for SCHISM:
      RH(%) + T(K) + P(Pa) -> spfh(kg/kg)
  - Precipitation is read from the MSM accumulated field (APCP-like local field).
    It is converted to precipitation rate (kg/m^2/s) by dividing by 3600.
  - The script reads GRIB with xarray+cfgrib.
  - By default, only the first 3 forecast hours from each 3-hourly cycle are used
    (instant fields: lead 0/1/2h, accum precip: lead 1/2/3h).
  - By default, processing is streamed month-by-month to reduce memory use.

Flags:
  --msm-root
      Root directory containing yearly MSM folders (YYYY/).
      Example: /path/to/msm_data where /path/to/msm_data/2012/*.bin exists.

  --start
      Start datetime for output period, inclusive.
      Format: "YYYY-MM-DD HH:MM:SS".

  --end
      End datetime for output period, inclusive.
      Format: "YYYY-MM-DD HH:MM:SS".

  --outdir
      Output directory for SCHISM files and config_used.json.

  --stack-hours
      Number of hours per output stack file.
      Example: 24 writes daily files, 72 writes 3-day files.

  --index-pad
      Zero-padding width of stack index in filenames.
      Example: 4 -> sflux_air_1.0001.nc, 0 -> sflux_air_1.1.nc.

  --overwrite
      If set, overwrite existing sflux files in --outdir.

  --output-lon-system
      Output longitude convention for sflux files:
        180 -> [-180, 180)
        360 -> [0, 360)

  --bbox LON_MIN LON_MAX LAT_MIN LAT_MAX
      Spatial subset before writing output.
      Example: --bbox 140 142 37 39

  --prc-missing-fill
      Fill value for missing precipitation-rate cells (prate).
      Default is 0.0.

  --config
      JSON file for bulk config override. CLI flags still override JSON values.

  --lead-hours-per-cycle
      Number of earliest forecast hours to keep from each GRIB cycle.
      Default: 3.

  --stream-mode {month,all}
      month: process and write month-by-month (memory efficient, default)
      all: process full START..END in a single pass

  --use-mpi
      Enable MPI parallel execution with mpirun/srun + mpi4py.
      Behavior:
        stream-mode=month -> months are distributed across ranks
        stream-mode=all   -> output stacks are distributed across ranks

Examples:
  1) Basic monthly run:
     python make_sflux_jma_msm.py \
       --msm-root /S/data00/G6008/d1041/dataset/JMA-MSM/msm_data \
       --start "2012-01-01 00:00:00" \
       --end   "2012-01-31 23:00:00" \
       --outdir ./sflux_jma_201201 \
       --stack-hours 24 \
       --index-pad 4

  2) Regional subset with overwrite:
     python make_sflux_jma_msm.py \
       --msm-root /S/data00/G6008/d1041/dataset/JMA-MSM/msm_data \
       --start "2012-06-01 00:00:00" \
       --end   "2012-06-30 23:00:00" \
       --outdir ./sflux_jma_sendai \
       --bbox 141 142 38 39 \
       --overwrite

  3) Use JSON config:
     python make_sflux_jma_msm.py --config config_jma_sflux.json

  4) MPI run (example with 8 ranks):
     mpirun -np 8 python make_sflux_jma_msm.py \
       --use-mpi \
       --msm-root /S/data00/G6008/d1041/dataset/JMA-MSM/msm_data \
       --start "2012-01-01 00:00:00" \
       --end   "2012-12-31 23:00:00" \
       --outdir ./sflux_jma_2012
"""

import argparse
import ctypes
import gc
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
if os.environ.get("MAKE_SFLUX_DISABLE_MPI", "0") == "1":
    COMM = None
    RANK = 0
    SIZE = 1
else:
    try:
        from mpi4py import MPI  # type: ignore
        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()
        SIZE = COMM.Get_size()
    except Exception:
        COMM = None
        RANK = 0
        SIZE = 1

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None


CONFIG = dict(
    MSM_ROOT="./msm_data",
    START="2012-01-01 00:00:00",
    END="2014-12-31 23:00:00",
    OUTDIR="./sflux_msm_2012to2014",
    STACK_HOURS=24,
    INDEX_PAD=0,
    OVERWRITE=False,
    OUTPUT_LON_SYSTEM="180",  # "180" or "360"
    BBOX=dict(lon_min=140.0, lon_max=142.0, lat_min=37.0, lat_max=39.0),
    PRC_MISSING_FILL=0.0,
    LEAD_HOURS_PER_CYCLE=3,
    STREAM_MODE="month",  # "month" or "all"
    USE_MPI=True,
    ISOLATE_STACK_PROCESS=True,
)


def log(msg, rank0_only=False):
    if rank0_only and RANK != 0:
        return
    prefix = f"[rank {RANK}/{SIZE}] " if SIZE > 1 else ""
    print(prefix + str(msg), flush=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_time(s):
    return pd.to_datetime(s)


def release_memory():
    gc.collect()
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def normalize_lon(lon, system):
    lon = np.asarray(lon, dtype=float)
    if system == "360":
        return lon % 360.0
    if system == "180":
        return ((lon + 180.0) % 360.0) - 180.0
    raise ValueError("OUTPUT_LON_SYSTEM must be '180' or '360'")


def apply_output_lon(da, output_system):
    if "lon" not in da.coords:
        return da
    lon = normalize_lon(da["lon"].values, output_system)
    da = da.assign_coords(lon=lon)
    if not np.all(np.diff(da["lon"].values) > 0):
        da = da.sortby("lon")
    return da


def subset_bbox(da, bbox):
    if bbox is None:
        return da
    if "lon" in da.coords:
        da = da.sel(lon=slice(float(bbox["lon_min"]), float(bbox["lon_max"])))
    if "lat" in da.coords:
        da = da.sel(lat=slice(float(bbox["lat_min"]), float(bbox["lat_max"])))
    return da


def open_grib(path, filter_by_keys):
    kwargs = dict(
        filter_by_keys=filter_by_keys,
        indexpath="",  # avoid stale *.idx mismatch across environments
    )
    try:
        return xr.open_dataset(path, engine="cfgrib", backend_kwargs=kwargs)
    except Exception:
        return None


def da_to_time_lat_lon(da):
    if "step" in da.dims and "valid_time" in da.coords:
        da = da.swap_dims({"step": "valid_time"})
        if "time" in da.coords and "time" not in da.dims:
            da = da.drop_vars("time")
        da = da.rename({"valid_time": "time"})
        if "step" in da.coords:
            da = da.drop_vars("step")
    rename = {}
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)
    # Remove scalar coords that are not needed in output
    for c in ("heightAboveGround", "surface", "meanSea"):
        if c in da.coords and c not in da.dims:
            da = da.drop_vars(c)
    if "lat" in da.coords and not np.all(np.diff(da["lat"].values) > 0):
        da = da.sortby("lat")
    return da


def concat_dedup_keep_last(das):
    if not das:
        return None
    out = xr.concat(
        das,
        dim="time",
        coords="minimal",
        compat="override",
        join="override",
    )
    out = out.sortby("time")
    t = pd.to_datetime(out["time"].values)
    keep = ~pd.Index(t).duplicated(keep="last")
    out = out.isel(time=np.where(keep)[0])
    return out


def calc_specific_humidity(rh_pct, temp_k, press_pa):
    # Tetens saturation vapor pressure over water
    temp_c = temp_k - 273.15
    es = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e = es * (rh_pct / 100.0)
    q = (0.622 * e) / np.maximum(press_pa - 0.378 * e, 1.0)
    return np.maximum(q, 0.0)


def parse_cycle_time_from_name(path):
    name = os.path.basename(path)
    m = re.search(r"_([0-9]{14})_MSM_GPV_", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def select_first_lead_hours(da, cycle_time, lead_hours, is_accum=False):
    """
    Keep only early lead times from one cycle.
      instant: 0 <= lead < lead_hours
      accum  : 0 <  lead <= lead_hours
    """
    if da is None or "time" not in da.coords:
        return da
    if cycle_time is None:
        return da

    c0 = np.datetime64(cycle_time)
    lead = (da["time"].values - c0) / np.timedelta64(1, "h")
    lead = np.asarray(lead, dtype=float)

    if is_accum:
        keep = (lead > 0.0) & (lead <= float(lead_hours))
    else:
        keep = (lead >= 0.0) & (lead < float(lead_hours))
    return da.isel(time=np.where(keep)[0])


def list_candidate_files(msm_root, start, end):
    # Include one previous cycle to capture accum field at start hour.
    begin = parse_time(start) - pd.Timedelta(hours=3)
    finish = parse_time(end)
    years = range(begin.year, finish.year + 1)
    files = []
    for y in years:
        ydir = os.path.join(msm_root, f"{y:04d}")
        if not os.path.isdir(ydir):
            continue
        files.extend(sorted(glob.glob(os.path.join(ydir, "*FH00-15_grib2.bin"))))

    out = []
    for fp in files:
        ctime = parse_cycle_time_from_name(fp)
        if ctime is None:
            continue
        if begin.to_pydatetime() <= ctime <= finish.to_pydatetime():
            out.append(fp)
    return sorted(out)


def build_cycle_catalog(msm_root, start, end):
    files = list_candidate_files(msm_root, start, end)
    out = []
    for fp in files:
        ctime = parse_cycle_time_from_name(fp)
        if ctime is not None:
            out.append((ctime, fp))
    return out


def select_cycle_files(catalog, start, end):
    begin = parse_time(start) - pd.Timedelta(hours=3)
    finish = parse_time(end)
    b = begin.to_pydatetime()
    f = finish.to_pydatetime()
    return [fp for ctime, fp in catalog if b <= ctime <= f]


def read_file_fields(path, t0, t1, bbox, out_lon_system, lead_hours_per_cycle):
    def finalize_da(da, is_accum=False):
        if da is None:
            return None
        da = select_first_lead_hours(
            da_to_time_lat_lon(da),
            cycle_time,
            lead_hours_per_cycle,
            is_accum=is_accum,
        )
        da = da.sel(time=slice(t0, t1))
        da = apply_output_lon(da, out_lon_system)
        da = subset_bbox(da, bbox)
        if "time" not in da.dims or da.sizes["time"] == 0:
            return None
        # Convert to plain numpy immediately to detach from cfgrib/xarray backend.
        tvals = pd.to_datetime(da["time"].values).to_numpy(dtype="datetime64[s]")
        lat = np.asarray(da["lat"].values)
        lon = np.asarray(da["lon"].values)
        vals = np.asarray(da.values, dtype=np.float32)
        return dict(time=tvals, lat=lat, lon=lon, data=vals)

    out = {}
    cycle_time = parse_cycle_time_from_name(path)

    # 10m winds
    ds_uv = open_grib(path, dict(stepType="instant", typeOfLevel="heightAboveGround", level=10))
    if ds_uv is not None:
        if "u10" in ds_uv:
            out["uwind"] = finalize_da(ds_uv["u10"], is_accum=False)
        if "v10" in ds_uv:
            out["vwind"] = finalize_da(ds_uv["v10"], is_accum=False)
        ds_uv.close()

    # 2m/1.5m T and RH (MSM is exposed by cfgrib at level=2 here)
    ds_tr = open_grib(path, dict(stepType="instant", typeOfLevel="heightAboveGround", level=2))
    if ds_tr is not None:
        if "t" in ds_tr:
            out["stmp"] = finalize_da(ds_tr["t"], is_accum=False)
        if "r" in ds_tr:
            out["rh"] = finalize_da(ds_tr["r"], is_accum=False)
        ds_tr.close()

    # Sea-level pressure (fallback to surface pressure)
    ds_p = open_grib(path, dict(stepType="instant", shortName="prmsl"))
    pvar = None
    if ds_p is not None and "prmsl" in ds_p:
        pvar = finalize_da(ds_p["prmsl"], is_accum=False)
        ds_p.close()
    else:
        if ds_p is not None:
            ds_p.close()
        ds_sp = open_grib(path, dict(stepType="instant", shortName="sp"))
        if ds_sp is not None and "sp" in ds_sp:
            pvar = finalize_da(ds_sp["sp"], is_accum=False)
            ds_sp.close()
        elif ds_sp is not None:
            ds_sp.close()
    if pvar is not None:
        out["prmsl"] = pvar

    # Accumulated precipitation (APCP-like local field)
    ds_acc = open_grib(path, dict(stepType="accum", typeOfLevel="surface"))
    if ds_acc is not None:
        for cand in ("apcp", "tp", "unknown"):
            if cand in ds_acc:
                out["apcp"] = finalize_da(ds_acc[cand], is_accum=True)
                break
        ds_acc.close()

    return out


def make_hourly_schedule(start, end):
    return pd.date_range(parse_time(start), parse_time(end), freq="1h")


def stack_slices(times, stack_hours):
    t = pd.to_datetime(times)
    if len(t) == 0:
        return []
    if len(t) == 1:
        return [slice(0, 1)]
    dt_h = np.median(np.diff(t).astype("timedelta64[s]").astype(float)) / 3600.0
    dt_h = max(float(dt_h), 1e-6)
    n = max(1, int(round(stack_hours / dt_h)))
    out = []
    i0 = 0
    while i0 < len(t):
        i1 = min(i0 + n, len(t))
        out.append(slice(i0, i1))
        i0 = i1
    return out


def time_base_and_values(times):
    t0 = pd.to_datetime(times[0]).to_pydatetime()
    base = datetime(t0.year, t0.month, t0.day)
    tvals = (pd.to_datetime(times) - base) / pd.Timedelta("1D")
    units = f"days since {base.strftime('%Y-%m-%d')} 00:00 UTC"
    base_date = [base.year, base.month, base.day, 0]
    return np.asarray(tvals, dtype=float), units, base_date


def format_index(idx, pad):
    if pad and pad > 0:
        return f"{idx:0{pad}d}"
    return str(idx)


def write_sflux(kind, idx, times, lon2, lat2, fields, outdir, overwrite=False, index_pad=0):
    idx_txt = format_index(idx, index_pad)
    fname = os.path.join(outdir, f"sflux_{kind}_1.{idx_txt}.nc")
    if os.path.exists(fname) and not overwrite:
        return fname, False

    tvals, tunits, base_date = time_base_and_values(times)
    nt = len(tvals)
    ny, nx = lon2.shape

    with Dataset(fname, "w") as nc:
        nc.createDimension("time", nt)
        nc.createDimension("nx_grid", nx)
        nc.createDimension("ny_grid", ny)

        vtime = nc.createVariable("time", "f8", ("time",))
        vtime.long_name = "Time"
        vtime.standard_name = "time"
        vtime.units = tunits
        vtime.base_date = base_date
        vtime[:] = tvals

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


def month_windows(start, end):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    cur = pd.Timestamp(year=s.year, month=s.month, day=1, hour=0, minute=0, second=0)
    out = []
    while cur <= e:
        mend = (cur + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)
        ws = max(cur, s)
        we = min(mend, e)
        if ws <= we:
            out.append((ws, we))
        cur = (cur + pd.offsets.MonthBegin(1)).replace(day=1, hour=0, minute=0, second=0)
    return out


def build_fields_for_interval(cfg, files, start, end):
    t0 = parse_time(start)
    t1 = parse_time(end)
    variables = ("uwind", "vwind", "stmp", "rh", "prmsl", "apcp")
    # Per-variable map of time -> 2D array; later files overwrite older cycles.
    collected = {k: {} for k in variables}
    lon_1d = None
    lat_1d = None

    if not files:
        raise RuntimeError(f"No MSM cycle files for stack window: {start} -> {end}")

    for fp in files:
        fields = read_file_fields(
            fp,
            t0,
            t1,
            cfg["BBOX"],
            cfg["OUTPUT_LON_SYSTEM"],
            int(cfg["LEAD_HOURS_PER_CYCLE"]),
        )
        for k, rec in fields.items():
            if k not in collected or rec is None:
                continue
            tvals = rec["time"]
            vals = rec["data"]
            if tvals.size == 0:
                continue

            # Save grid from first valid variable.
            if lon_1d is None or lat_1d is None:
                lon_1d = np.asarray(rec["lon"])
                lat_1d = np.asarray(rec["lat"])

            if vals.ndim != 3:
                raise RuntimeError(f"Unexpected ndim for variable '{k}': {vals.ndim}")
            for it, tv in enumerate(tvals):
                collected[k][tv] = vals[it]
        del fields
        release_memory()
    release_memory()

    needed_air = ("uwind", "vwind", "stmp", "rh", "prmsl")
    for k in needed_air:
        if not collected[k]:
            raise RuntimeError(f"Missing required variable in stack window: {k}")

    out_times = make_hourly_schedule(start, end)
    if len(out_times) == 0:
        raise RuntimeError(f"No output times in stack window: {start} -> {end}")

    if lon_1d is None or lat_1d is None:
        raise RuntimeError("Failed to determine MSM lon/lat grid for stack window.")
    ny = len(lat_1d)
    nx = len(lon_1d)
    out_t = pd.to_datetime(out_times).to_numpy(dtype="datetime64[s]")

    for k in needed_air:
        arr = np.full((len(out_t), ny, nx), np.nan, dtype=np.float32)
        cmap = collected[k]
        for it, tv in enumerate(out_t):
            vv = cmap.get(tv)
            if vv is not None:
                arr[it, :, :] = vv
        if np.isnan(arr).any():
            nmiss = int(np.isnan(arr).sum())
            raise RuntimeError(
                f"Variable '{k}' has {nmiss} missing values after hourly alignment. "
                "Check input completeness or adjust START/END."
            )
        collected[k] = arr

    spfh_vals = calc_specific_humidity(
        collected["rh"],
        collected["stmp"],
        collected["prmsl"],
    )

    prate_vals = np.full(
        (len(out_t), ny, nx), float(cfg["PRC_MISSING_FILL"]), dtype=np.float32
    )
    for it, tv in enumerate(out_t):
        vv = collected["apcp"].get(tv)
        if vv is not None:
            prate_vals[it, :, :] = vv / np.float32(3600.0)

    lon2, lat2 = np.meshgrid(lon_1d, lat_1d)
    air_fields = dict(
        prmsl=collected["prmsl"],
        spfh=np.asarray(spfh_vals, dtype=np.float32),
        stmp=collected["stmp"],
        uwind=collected["uwind"],
        vwind=collected["vwind"],
    )
    prc_fields = dict(
        prate=prate_vals,
    )
    # Release maps/temps now that plain numpy arrays are prepared.
    del collected, spfh_vals, prate_vals
    release_memory()
    return out_times, lon2, lat2, air_fields, prc_fields


def process_single_stack(cfg, idx, s0, s1, catalog=None):
    if catalog is None:
        catalog = build_cycle_catalog(cfg["MSM_ROOT"], s0, s1)
    stack_files = select_cycle_files(catalog, s0, s1)
    log(
        f"Stack {idx}: {s0} -> {s1}, "
        f"using {len(stack_files)} cycle files"
    )
    out_times, lon2, lat2, air_fields, prc_fields = build_fields_for_interval(
        cfg, stack_files, s0, s1
    )

    fair, _ = write_sflux(
        "air", idx, out_times, lon2, lat2, air_fields, cfg["OUTDIR"],
        overwrite=cfg["OVERWRITE"], index_pad=int(cfg["INDEX_PAD"])
    )
    fprc, _ = write_sflux(
        "prc", idx, out_times, lon2, lat2, prc_fields, cfg["OUTDIR"],
        overwrite=cfg["OVERWRITE"], index_pad=int(cfg["INDEX_PAD"])
    )
    log(f"Wrote stack {idx}: {fair}, {fprc}")
    del out_times, lon2, lat2, air_fields, prc_fields
    release_memory()


def run_stack_subprocess(cfg, idx, s0, s1):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--msm-root", str(cfg["MSM_ROOT"]),
        "--start", str(cfg["START"]),
        "--end", str(cfg["END"]),
        "--outdir", str(cfg["OUTDIR"]),
        "--stack-hours", str(int(cfg["STACK_HOURS"])),
        "--index-pad", str(int(cfg["INDEX_PAD"])),
        "--output-lon-system", str(cfg["OUTPUT_LON_SYSTEM"]),
        "--prc-missing-fill", str(float(cfg["PRC_MISSING_FILL"])),
        "--lead-hours-per-cycle", str(int(cfg["LEAD_HOURS_PER_CYCLE"])),
        "--stream-mode", "all",
        "--single-stack-start", s0,
        "--single-stack-end", s1,
        "--single-stack-index", str(int(idx)),
    ]
    bbox = cfg.get("BBOX", None)
    if bbox:
        cmd.extend([
            "--bbox",
            str(float(bbox["lon_min"])),
            str(float(bbox["lon_max"])),
            str(float(bbox["lat_min"])),
            str(float(bbox["lat_max"])),
        ])
    if cfg.get("OVERWRITE", False):
        cmd.append("--overwrite")

    env = os.environ.copy()
    env["MAKE_SFLUX_DISABLE_MPI"] = "1"
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Stack subprocess failed for idx={idx}, {s0} -> {s1}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if proc.stdout.strip():
        for line in proc.stdout.strip().splitlines():
            log(line)


def process_window(cfg, start, end, file_index_start=1, stack_rank=0, stack_size=1):
    out_times_window = make_hourly_schedule(start, end)
    if len(out_times_window) == 0:
        log(f"No output times for window: {start} -> {end}. Skip.")
        return file_index_start

    slices = stack_slices(out_times_window, int(cfg["STACK_HOURS"]))
    if not slices:
        log(f"No stack slices for window: {start} -> {end}. Skip.")
        return file_index_start

    catalog = build_cycle_catalog(cfg["MSM_ROOT"], start, end)
    if not catalog:
        log(f"No MSM files found for window: {start} -> {end}. Skip.")
        return file_index_start
    log(f"Found {len(catalog)} MSM cycle files for {start} -> {end}")

    for i, slc in enumerate(slices):
        if stack_size > 1 and (i % stack_size) != stack_rank:
            continue
        idx = file_index_start + i
        tt = out_times_window[slc]
        s0 = pd.to_datetime(tt[0]).strftime("%Y-%m-%d %H:%M:%S")
        s1 = pd.to_datetime(tt[-1]).strftime("%Y-%m-%d %H:%M:%S")
        if bool(cfg.get("ISOLATE_STACK_PROCESS", False)):
            run_stack_subprocess(cfg, idx, s0, s1)
        else:
            process_single_stack(cfg, idx, s0, s1, catalog=catalog)
    return file_index_start + len(slices)


def parse_args(argv=None):
    class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    epilog = (
        "Examples:\n"
        "  python make_sflux_jma_msm.py --msm-root ./msm_data \\\n"
        "    --start '2012-01-01 00:00:00' --end '2012-01-31 23:00:00' \\\n"
        "    --outdir ./sflux_jma --stack-hours 24 --index-pad 4 \\\n"
        "    --lead-hours-per-cycle 3 --stream-mode month\n\n"
        "  mpirun -np 8 python make_sflux_jma_msm.py --use-mpi \\\n"
        "    --msm-root ./msm_data --start '2012-01-01 00:00:00' \\\n"
        "    --end '2012-12-31 23:00:00' --outdir ./sflux_jma_2012\n\n"
        "  python make_sflux_jma_msm.py --config config_jma_sflux.json"
    )

    ap = argparse.ArgumentParser(
        description="Create SCHISM sflux_air/sflux_prc from JMA MSM GRIB2.",
        formatter_class=_Formatter,
        epilog=epilog,
    )
    ap.add_argument(
        "--msm-root",
        help="Root directory containing yearly MSM GRIB folders (YYYY/).",
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
        "--outdir",
        help="Output directory for sflux files and config_used.json.",
    )
    ap.add_argument(
        "--stack-hours",
        type=int,
        help="Hours per output stack file (e.g., 24 for daily files).",
    )
    ap.add_argument(
        "--index-pad",
        type=int,
        help="Zero-padding width for stack index in output filenames.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files in --outdir.",
    )
    ap.add_argument(
        "--output-lon-system",
        choices=("180", "360"),
        help="Output longitude convention: 180 for [-180,180), 360 for [0,360).",
    )
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Spatial subset bounds.",
    )
    ap.add_argument(
        "--prc-missing-fill",
        type=float,
        help="Fill value for missing precipitation-rate cells.",
    )
    ap.add_argument(
        "--lead-hours-per-cycle",
        type=int,
        help="Use only earliest N lead hours per cycle (default 3).",
    )
    ap.add_argument(
        "--stream-mode",
        choices=("month", "all"),
        help="Streaming mode: month (memory efficient) or all (single window).",
    )
    ap.add_argument(
        "--isolate-stack-process",
        action="store_true",
        help="Run each stack in a subprocess (slower, but avoids cfgrib/eccodes memory growth).",
    )
    ap.add_argument(
        "--no-isolate-stack-process",
        action="store_true",
        help="Disable per-stack subprocess isolation (isolation is enabled by default).",
    )
    ap.add_argument(
        "--use-mpi",
        action="store_true",
        help="Enable MPI parallel run (requires mpi4py and mpirun/srun).",
    )
    ap.add_argument(
        "--config",
        help="Optional JSON config file. CLI flags override JSON values.",
    )
    ap.add_argument("--single-stack-start", help=argparse.SUPPRESS)
    ap.add_argument("--single-stack-end", help=argparse.SUPPRESS)
    ap.add_argument("--single-stack-index", type=int, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = dict(CONFIG)

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))

    if args.msm_root:
        cfg["MSM_ROOT"] = args.msm_root
    if args.start:
        cfg["START"] = args.start
    if args.end:
        cfg["END"] = args.end
    if args.outdir:
        cfg["OUTDIR"] = args.outdir
    if args.stack_hours is not None:
        cfg["STACK_HOURS"] = int(args.stack_hours)
    if args.index_pad is not None:
        cfg["INDEX_PAD"] = int(args.index_pad)
    if args.overwrite:
        cfg["OVERWRITE"] = True
    if args.output_lon_system:
        cfg["OUTPUT_LON_SYSTEM"] = args.output_lon_system
    if args.bbox:
        cfg["BBOX"] = dict(
            lon_min=float(args.bbox[0]),
            lon_max=float(args.bbox[1]),
            lat_min=float(args.bbox[2]),
            lat_max=float(args.bbox[3]),
        )
    if args.prc_missing_fill is not None:
        cfg["PRC_MISSING_FILL"] = float(args.prc_missing_fill)
    if args.lead_hours_per_cycle is not None:
        cfg["LEAD_HOURS_PER_CYCLE"] = int(args.lead_hours_per_cycle)
    if args.stream_mode:
        cfg["STREAM_MODE"] = args.stream_mode
    if args.isolate_stack_process:
        cfg["ISOLATE_STACK_PROCESS"] = True
    if args.no_isolate_stack_process:
        cfg["ISOLATE_STACK_PROCESS"] = False
    if args.use_mpi:
        cfg["USE_MPI"] = True

    if SIZE > 1 and not cfg["USE_MPI"]:
        log("MPI ranks detected; enabling MPI mode automatically.", rank0_only=True)
        cfg["USE_MPI"] = True
    use_mpi = bool(cfg["USE_MPI"]) and SIZE > 1

    if args.single_stack_start is not None:
        if args.single_stack_end is None or args.single_stack_index is None:
            raise RuntimeError(
                "--single-stack-start requires --single-stack-end and --single-stack-index"
            )
        ensure_dir(cfg["OUTDIR"])
        process_single_stack(
            cfg,
            int(args.single_stack_index),
            str(args.single_stack_start),
            str(args.single_stack_end),
            catalog=None,
        )
        return

    if RANK == 0:
        ensure_dir(cfg["OUTDIR"])
        with open(os.path.join(cfg["OUTDIR"], "config_used.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    if use_mpi and COMM is not None:
        COMM.Barrier()

    if cfg["STREAM_MODE"] == "month":
        windows = month_windows(cfg["START"], cfg["END"])

        # Precompute deterministic file index offsets by time window.
        starts = []
        idx = 1
        for ws, we in windows:
            times = make_hourly_schedule(
                ws.strftime("%Y-%m-%d %H:%M:%S"),
                we.strftime("%Y-%m-%d %H:%M:%S"),
            )
            nstacks = len(stack_slices(times, int(cfg["STACK_HOURS"])))
            starts.append(idx)
            idx += nstacks

        for i, (ws, we) in enumerate(windows):
            process_window(
                cfg,
                ws.strftime("%Y-%m-%d %H:%M:%S"),
                we.strftime("%Y-%m-%d %H:%M:%S"),
                file_index_start=starts[i],
                stack_rank=(RANK if use_mpi else 0),
                stack_size=(SIZE if use_mpi else 1),
            )
    else:
        if use_mpi:
            process_window(
                cfg,
                cfg["START"],
                cfg["END"],
                file_index_start=1,
                stack_rank=RANK,
                stack_size=SIZE,
            )
        else:
            process_window(cfg, cfg["START"], cfg["END"], file_index_start=1)

    if use_mpi and COMM is not None:
        COMM.Barrier()
    log("Done.", rank0_only=True)


if __name__ == "__main__":
    main()
