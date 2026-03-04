#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCHISM 2D Map Maker — integrated, optimized, and MPI-enabled
with temporal aggregation: --tmean {none,hourly,daily,monthly}
"""

# =====================
# Config
# =====================
USER_CONFIG = {
    "enable": True,

    # --- Experiment-driven paths ---
    "exp": "RUN04a",                                # e.g., RUN01a/RUN01b/...
    "base_run_dir": "/scratch2/08924/kmpark/SOB/run/", # parent dir for runs
    "base_outdir": "./images/",                      # parent dir for images

    # If any are None, they'll be derived from exp:
    "hgrid": None,   # -> {base_run_dir}/{exp}/hgrid.gr3
    "run": None,     # -> {base_run_dir}/{exp}/outputs
    "outdir": None,  # -> {base_outdir}/{exp}

    # Range & time
    "start": None,             # None -> auto from discovered stacks
    "end": None,               # None -> auto from discovered stacks
    "refdate": "2017-01-02", # It is not used when "apply_param_start_time": True
    "apply_param_start_time": True,
    "apply_utc_start": False,
    "param_nml": None,          # if None: auto-detect from run dir

    # Layer & intra-file aggregation
    "layer": -1,              # -1 = surface
    "agg": "inst",            # file-mean or inst (inst -> all steps when tmean=none)

    # Layout & variables
    "layout": "ALL",          # EC, TS, ALL, SINGLE
    "vars": None,

    # Plot
    "prefix": "",
    "dpi": 500,
    "skip_existing": False,    # True: skip if output PNG already exists
    "cbars": "each",          # each, shared, none
    "proj": "PlateCarree",
    "extent": None,           # [xmin xmax ymin ymax]
    "coastline": False,
    "bnd": False,
    "bnd_color": "k",
    "bnd_lw": 0.6,

    # MPI & temporal mean
    "mpi": True,
    "tmean": "none",          # none | hourly | daily | monthly
    "mpi_show_assignment": True,
    "mpi_log_every": 1,         # print rank progress every N local stacks
    "mpi_rank_logs": False,     # write per-rank logs in outdir

    # Reporting
    "report_stack_info": False,
    "stack_span_hours": None,

    # Stack screening
    "stack_check_mode": "light",        # none | light | size | light+size
    "stack_check_all_files": False,     # False: check primary file only
    "stack_size_ratio_min": 0.70,       # for size/light+size
    "stack_size_min_bytes": None,       # optional absolute size floor

    # Temporal-mean accumulator dtype
    "accum_dtype": "float32", # float32 | float64
}

# ---- headless & cluster guards ----
import os
os.environ.setdefault("MPLBACKEND", "Agg")
if "frontera" in (os.environ.get("HOSTNAME", "").lower(), os.environ.get("TACC_SYSTEM", "").lower()):
    os.environ["TACC_SYSTEM"] = "headless"

# ---- std libs ----
import sys
import argparse
from pathlib import Path
import re
from glob import glob
import numpy as np
import gc
import time

# ---- plotting ----
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cmocean

# ---- netCDF (memory-lean I/O) ----
from netCDF4 import Dataset

# ---- SCHISM helpers ----
from pylib import read_schism_hgrid, read_schism_param, datenum, num2date

# ---- MPI (optional) ----
try:
    from mpi4py import MPI  # type: ignore
    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
except Exception:
    _COMM = None
    _RANK = 0
    _SIZE = 1

# -------------------------
# Helpers
# -------------------------
VAR_DEFAULTS = {
    "ssh":  {"cmap": "cmocean.balance", "clim": (-1.5, 1.5), "title": "Sea Surface Height (m)"},
    "uv":   {"cmap": "cmocean.speed",   "clim": (0.0, 1.0),   "title": "Surface Current Speed (m s$^{-1}$)"},
    "temp": {"cmap": "cmocean.thermal", "clim": (7.0, 15.0),  "title": "Temperature (°C)"},
    "salt": {"cmap": "cmocean.haline",  "clim": (32.0, 34.0),  "title": "Salinity (psu)"},
}

LAYOUT_TO_VARS = {
    "EC":   ["ssh", "uv"],
    "TS":   ["temp", "salt"],
    "ALL":  ["ssh", "uv", "temp", "salt"],
}

def _get_cmap(name: str):
    try:
        if name.startswith("cmocean."):
            return getattr(cmocean.cm, name.split(".", 1)[1])
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap("viridis")

def _as_f32_with_nan(a):
    """Return a writeable float32 ndarray; masked values -> NaN."""
    if np.ma.isMaskedArray(a):
        return np.ma.filled(a, np.nan).astype(np.float32, copy=False)
    return np.asarray(a, dtype=np.float32)

def _as_f64_with_nan(a):
    """Return a writeable float64 ndarray; masked values -> NaN."""
    if np.ma.isMaskedArray(a):
        return np.ma.filled(a, np.nan).astype(np.float64, copy=False)
    return np.asarray(a, dtype=np.float64)

def _mod360(lon):
    return np.mod(np.asarray(lon), 360.0)

def _wrap_to_axis(lon, central_lon):
    """Shift to axes CRS: (-180,180] relative to central_lon."""
    return ((np.asarray(lon) - central_lon + 180.0) % 360.0) - 180.0

def _choose_central_longitude(lon0_360):
    """
    Pick central_longitude so the domain doesn't cross the map edge:
    cut the circle at the largest empty gap between sorted longitudes.
    """
    lon = np.sort(_mod360(lon0_360))
    if lon.size == 0:
        return 0.0
    diffs = np.diff(lon)
    wrap_gap = lon[0] + 360.0 - lon[-1]
    diffs = np.append(diffs, wrap_gap)
    i = int(np.argmax(diffs))
    left = lon[i]
    right = lon[(i + 1) % lon.size]
    central = (left + right) / 2.0 % 360.0
    return float(central)

def parse_args():
    p = argparse.ArgumentParser(description="Integrated 2D map generator for SCHISM outputs")
    # Experiment-driven
    p.add_argument("--exp")
    p.add_argument("--base-run-dir")
    p.add_argument("--base-outdir")
    # Inputs
    p.add_argument("--hgrid")
    p.add_argument("--run")
    p.add_argument("--outdir")
    p.add_argument("--start", type=int)
    p.add_argument("--end",   type=int)
    p.add_argument("--refdate")
    p.add_argument("--param-nml")
    p.add_argument("--apply-param-start-time", dest="apply_param_start_time", action="store_true")
    p.add_argument("--no-apply-param-start-time", dest="apply_param_start_time", action="store_false")
    p.add_argument("--apply-utc-start", dest="apply_utc_start", action="store_true")
    p.add_argument("--no-apply-utc-start", dest="apply_utc_start", action="store_false")
    p.add_argument("--layer", type=int)
    p.add_argument("--agg", choices=["file-mean", "inst"])
    p.add_argument("--layout", choices=["EC", "TS", "ALL", "SINGLE"])
    p.add_argument("--vars", nargs="+", choices=list(VAR_DEFAULTS.keys()))
    p.add_argument("--prefix")
    p.add_argument("--dpi", type=int)
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--cbars", choices=["each", "shared", "none"])
    p.add_argument("--proj")
    p.add_argument("--extent", nargs=4, type=float)
    p.add_argument("--no-coastline", action="store_true")
    p.add_argument("--bnd", action="store_true")
    p.add_argument("--bnd-color")
    p.add_argument("--bnd-lw", type=float)
    # Temporal mean
    p.add_argument("--tmean", choices=["none", "hourly", "daily", "monthly"])
    # MPI observability
    p.add_argument("--mpi-show-assignment", dest="mpi_show_assignment", action="store_true")
    p.add_argument("--no-mpi-show-assignment", dest="mpi_show_assignment", action="store_false")
    p.add_argument("--mpi-log-every", type=int)
    p.add_argument("--mpi-rank-logs", dest="mpi_rank_logs", action="store_true")
    p.add_argument("--no-mpi-rank-logs", dest="mpi_rank_logs", action="store_false")
    # Reporting
    p.add_argument("--report-stack-info", action="store_true")
    p.add_argument("--stack-span-hours", type=float)
    p.add_argument("--stack-check-mode", choices=["none", "light", "size", "light+size"])
    p.add_argument("--stack-check-all-files", dest="stack_check_all_files", action="store_true")
    p.add_argument("--no-stack-check-all-files", dest="stack_check_all_files", action="store_false")
    p.add_argument("--stack-size-ratio-min", type=float)
    p.add_argument("--stack-size-min-bytes", type=int)
    # Accumulator dtype
    p.add_argument("--accum-dtype", choices=["float32", "float64"])
    # Toggles
    p.add_argument("--no-config", action="store_true")
    p.add_argument("--no-mpi", action="store_true")
    p.set_defaults(
        apply_param_start_time=None,
        apply_utc_start=None,
        stack_check_all_files=None,
        mpi_show_assignment=None,
        mpi_rank_logs=None,
        skip_existing=None,
    )
    return p.parse_args()

def get_projection(name: str, central_longitude: float = 0.0):
    if name == "PlateCarree":
        return ccrs.PlateCarree(central_longitude=central_longitude)
    if name == "Mercator":
        return ccrs.Mercator()
    if name == "LambertConformal":
        return ccrs.LambertConformal()
    return ccrs.PlateCarree(central_longitude=central_longitude)

def _to_scalar(v, default=None):
    if v is None:
        return default
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            return default
        return v[0]
    return v

def _resolve_run_root(run_path):
    p = Path(run_path).resolve()
    return p.parent if p.name == "outputs" else p

def _get_model_start_datenum(run_path, param_nml=None, apply_utc_start=False):
    if param_nml:
        pfile = Path(param_nml)
    else:
        pfile = _resolve_run_root(run_path) / "param.nml"

    if not pfile.exists():
        return None, f"param.nml not found: {pfile}"

    try:
        p = read_schism_param(str(pfile), 1)
    except Exception as exc:
        return None, f"failed to parse {pfile}: {exc}"

    keys = ["start_year", "start_month", "start_day", "start_hour"]
    for key in keys:
        if key not in p:
            return None, f"missing {key} in {pfile}"

    try:
        sy = int(_to_scalar(p.get("start_year")))
        sm = int(_to_scalar(p.get("start_month")))
        sd = int(_to_scalar(p.get("start_day")))
        sh = float(_to_scalar(p.get("start_hour"), 0.0))
        us = float(_to_scalar(p.get("utc_start"), 0.0))
    except Exception as exc:
        return None, f"invalid start time fields in {pfile}: {exc}"

    d0 = float(datenum(sy, sm, sd))
    d0 = d0 + sh / 24.0
    if apply_utc_start:
        d0 = d0 - us / 24.0

    info = f"{pfile} -> {sy:04d}-{sm:02d}-{sd:02d} {sh:05.2f}h (utc_start={us})"
    return d0, info

def _stack_num_from_name(path):
    m = re.search(r"_(\d+)\.nc$", os.path.basename(path))
    return int(m.group(1)) if m else None

def _required_stack_files(run_dir, stack, vars_to_plot):
    templates = []
    for var in vars_to_plot:
        if var == "ssh":
            templates.extend(["out2d_{stack}.nc"])
        elif var == "uv":
            templates.extend(["horizontalVelX_{stack}.nc", "horizontalVelY_{stack}.nc"])
        elif var == "temp":
            templates.extend(["temperature_{stack}.nc"])
        elif var == "salt":
            templates.extend(["salinity_{stack}.nc"])
    uniq = []
    seen = set()
    for t in templates:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return [os.path.join(run_dir, t.format(stack=int(stack))) for t in uniq]

def _primary_stack_file(run_dir, stack, vars_to_plot):
    files = _required_stack_files(run_dir, stack, vars_to_plot)
    if len(files) == 0:
        return None
    out2d = os.path.join(run_dir, f"out2d_{int(stack)}.nc")
    if out2d in files:
        return out2d
    return files[0]

def _header_time_ok(nc_path):
    try:
        with Dataset(nc_path, "r") as nc:
            if "time" not in nc.variables:
                return False, "missing time variable"
            tvar = nc.variables["time"]
            nt = int(tvar.shape[0]) if hasattr(tvar, "shape") and len(tvar.shape) > 0 else int(len(np.array(tvar)))
            if nt <= 0:
                return False, "empty time variable"
            _ = float(np.array(tvar[0]).ravel()[0])
        return True, "ok"
    except Exception as exc:
        return False, str(exc)

def _size_ok(path, ref_size, ratio_min=0.70, abs_min_bytes=None):
    try:
        size = int(os.path.getsize(path))
    except Exception as exc:
        return False, f"size check failed: {exc}"
    if abs_min_bytes is not None and size < int(abs_min_bytes):
        return False, f"size={size} < abs_min={int(abs_min_bytes)}"
    thr = float(ratio_min) * float(ref_size)
    if size < thr:
        return False, f"size={size} < ratio_min*median={int(thr)}"
    return True, "ok"

def _screen_stacks(run_dir, stacks, vars_to_plot, mode="light", check_all_files=False,
                   ratio_min=0.70, abs_min_bytes=None):
    stacks = [int(i) for i in np.asarray(stacks).ravel()]
    if len(stacks) == 0:
        return np.array([], dtype=int), {}

    mode = "none" if mode is None else str(mode).lower()
    primary = {}
    for st in stacks:
        p = _primary_stack_file(run_dir, st, vars_to_plot)
        if p is not None and os.path.exists(p):
            primary[st] = p

    ref_size = None
    sizes = [os.path.getsize(fp) for fp in primary.values() if os.path.exists(fp)]
    if len(sizes) > 0:
        ref_size = int(np.median(np.asarray(sizes, dtype=float)))

    valid = []
    skipped = {}
    for st in stacks:
        req_files = _required_stack_files(run_dir, st, vars_to_plot)
        if len(req_files) == 0:
            skipped[st] = "no required files for selected variables"
            continue

        missing = [os.path.basename(fp) for fp in req_files if not os.path.exists(fp)]
        if len(missing) > 0:
            show = ", ".join(missing[:4])
            if len(missing) > 4:
                show += f", ... (+{len(missing) - 4} more)"
            skipped[st] = f"missing files: {show}"
            continue

        primary_file = _primary_stack_file(run_dir, st, vars_to_plot)
        files = req_files if check_all_files else [primary_file]

        need_light = mode in {"light", "light+size"}
        need_size = mode in {"size", "light+size"}
        ok = True
        reason = ""

        if need_light:
            for fn in files:
                f_ok, f_reason = _header_time_ok(fn)
                if not f_ok:
                    ok = False
                    reason = f"{os.path.basename(fn)}: {f_reason}"
                    break

        if ok and need_size and ref_size is not None:
            for fn in files:
                s_ok, s_reason = _size_ok(
                    fn,
                    ref_size,
                    ratio_min=ratio_min,
                    abs_min_bytes=abs_min_bytes,
                )
                if not s_ok:
                    ok = False
                    reason = f"{os.path.basename(fn)}: {s_reason}"
                    break

        if ok:
            valid.append(st)
        else:
            skipped[st] = reason if reason else "stack check failed"

    return np.asarray(valid, dtype=int), skipped

def _discover_stacks(run_dir):
    pats = [
        "out2d_*.nc",
        "horizontalVelX_*.nc",
        "horizontalVelY_*.nc",
        "temperature_*.nc",
        "salinity_*.nc",
    ]
    stacks = set()
    for pat in pats:
        for fn in glob(os.path.join(run_dir, pat)):
            st = _stack_num_from_name(fn)
            if st is not None:
                stacks.add(int(st))
    return np.asarray(sorted(stacks), dtype=int)

def _select_candidate_stacks(run_dir, start=None, end=None):
    avail = _discover_stacks(run_dir)
    if len(avail) == 0:
        return np.asarray([], dtype=int)

    lo = int(np.min(avail)) if start is None else int(start)
    hi = int(np.max(avail)) if end is None else int(end)
    if lo > hi:
        lo, hi = hi, lo
    return avail[(avail >= lo) & (avail <= hi)].astype(int)

def _get_time_origin_datenum(args):
    if bool(getattr(args, "apply_param_start_time", True)):
        dn, info = _get_model_start_datenum(
            args.run,
            param_nml=getattr(args, "param_nml", None),
            apply_utc_start=bool(getattr(args, "apply_utc_start", False)),
        )
        if dn is not None:
            return float(dn), f"param.nml ({info})"
        return float(datenum(args.refdate)), f"fallback refdate={args.refdate} (param parse failed: {info})"
    return float(datenum(args.refdate)), f"refdate={args.refdate}"

def read_time_and_mask_nc(nc, time_origin_dnum):
    t = _as_f64_with_nan(nc.variables["time"][:])  # seconds since ref
    ctime = t / 86400.0 + time_origin_dnum
    mvar = nc.variables.get("dryFlagNode")
    mask = _as_f32_with_nan(mvar[:]) if mvar is not None else None
    return np.asarray(ctime, dtype=np.float64), (np.asarray(mask) if mask is not None else None)

# -------- Memory-lean readers (single-layer) --------
def read_field(run_dir, var, istack, layer_idx, time_origin_dnum, out2d_for_mask=True):
    if var == "ssh":
        p = f"{run_dir}/out2d_{istack}.nc"
        with Dataset(p, "r") as nc:
            ctime, mask = read_time_and_mask_nc(nc, time_origin_dnum)
            elev = _as_f32_with_nan(nc.variables["elevation"][:])  # [time,node]
        return ctime, elev, mask

    if var == "uv":
        up = f"{run_dir}/horizontalVelX_{istack}.nc"
        vp = f"{run_dir}/horizontalVelY_{istack}.nc"
        with Dataset(up, "r") as ncu, Dataset(vp, "r") as ncv:
            ctime, mask = read_time_and_mask_nc(ncu, time_origin_dnum)
            u2 = _as_f32_with_nan(ncu.variables["horizontalVelX"][:, :, layer_idx])
            v2 = _as_f32_with_nan(ncv.variables["horizontalVelY"][:, :, layer_idx])
        speed = np.hypot(u2, v2).astype(np.float32, copy=False)
        del u2, v2
        if mask is None and out2d_for_mask:
            try:
                with Dataset(f"{run_dir}/out2d_{istack}.nc", "r") as nc2d:
                    m2 = nc2d.variables.get("dryFlagNode")
                    mask = (_as_f32_with_nan(m2[:]) if m2 is not None else None)
            except Exception:
                mask = None
        return ctime, speed, (np.asarray(mask) if mask is not None else None)

    if var in ("temp", "salt"):
        vname = "temperature" if var == "temp" else "salinity"
        path = f"{run_dir}/{vname}_{istack}.nc"
        with Dataset(path, "r") as nc:
            ctime, mask = read_time_and_mask_nc(nc, time_origin_dnum)
            lay = _as_f32_with_nan(nc.variables[vname][:, :, layer_idx])  # [time,node]
        if mask is None:
            try:
                with Dataset(f"{run_dir}/out2d_{istack}.nc", "r") as nc2d:
                    m2 = nc2d.variables.get("dryFlagNode")
                    mask = (_as_f32_with_nan(m2[:]) if m2 is not None else None)
            except Exception:
                mask = None
        return ctime, lay, (np.asarray(mask) if mask is not None else None)

    raise ValueError(f"Unknown var: {var}")

def reduce_time(arr_2d, agg):
    if agg == "file-mean":
        return np.nanmean(arr_2d, axis=0)
    return arr_2d[0, :]

def make_axes(n_panels, proj, layout):
    if layout == "SINGLE" or n_panels == 1:
        fig = plt.figure(figsize=(7.0, 6.0), constrained_layout=True)
        ax = plt.axes(projection=proj)
        return fig, [ax]
    if n_panels == 2:
        fig = plt.figure(figsize=(10.5, 8.0), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        axes = [fig.add_subplot(gs[0, i], projection=proj) for i in range(2)]
        return fig, axes
    fig = plt.figure(figsize=(12.0, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(gs[0, 0], projection=proj),
        fig.add_subplot(gs[0, 1], projection=proj),
        fig.add_subplot(gs[1, 0], projection=proj),
        fig.add_subplot(gs[1, 1], projection=proj),
    ]
    return fig, axes

def format_geoaxes(ax, gd, extent, title_txt=None, data_crs=None,
                   show_coastline=True, show_bnd=False,
                   bnd_color="k", bnd_lw=0.6):
    if show_coastline:
        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="0.9", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.98", zorder=0)
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".1f", zero_direction_label=True,
                                                    dateline_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".1f"))

    if data_crs is None:
        data_crs = ccrs.PlateCarree()

    if extent is None:
        ax.set_extent([float(np.nanmin(gd.x)), float(np.nanmax(gd.x)),
                       float(np.nanmin(gd.y)), float(np.nanmax(gd.y))],
                      crs=data_crs)
    else:
        ax.set_extent(extent, crs=data_crs)

    ax.set_xlabel("Longitude", labelpad=6, fontsize=8)
    ax.set_ylabel("Latitude",  labelpad=6, fontsize=8)

    if title_txt:
        ax.set_title(title_txt, fontsize=9, weight="bold")
    if show_bnd:
        try:
            gd.plot_bnd(ax=ax, color=bnd_color, lw=bnd_lw)
        except Exception:
            gd.plot_bnd(ax=ax)

def _merge_config(args):
    ad = vars(args).copy()
    use_cfg = USER_CONFIG.get("enable", False) and not ad.get("no_config", False)

    if use_cfg:
        for k, v in USER_CONFIG.items():
            if k == "enable":
                continue
            if ad.get(k) is None and v is not None:
                ad[k] = v

    exp = ad.get("exp")
    base_run_dir = ad.get("base_run_dir", USER_CONFIG.get("base_run_dir"))
    base_outdir  = ad.get("base_outdir",  USER_CONFIG.get("base_outdir"))

    if exp:
        if ad.get("hgrid") is None and base_run_dir:
            ad["hgrid"] = str(Path(base_run_dir) / exp / "hgrid.gr3")
        if ad.get("run") is None and base_run_dir:
            ad["run"] = str(Path(base_run_dir) / exp / "outputs")
        if ad.get("outdir") is None and base_outdir:
            ad["outdir"] = str(Path(base_outdir) / exp)

    # defaults
    ad.setdefault("refdate", "2022-01-02")
    ad.setdefault("apply_param_start_time", USER_CONFIG.get("apply_param_start_time", True))
    ad.setdefault("apply_utc_start", USER_CONFIG.get("apply_utc_start", False))
    ad.setdefault("param_nml", USER_CONFIG.get("param_nml", None))
    ad.setdefault("layer", -1)
    ad.setdefault("agg", "inst")
    ad.setdefault("layout", "EC")
    ad.setdefault("dpi", 200)
    ad.setdefault("skip_existing", USER_CONFIG.get("skip_existing", False))
    ad.setdefault("cbars", "each")
    ad.setdefault("proj", "PlateCarree")
    if ad.get("no_coastline"):
        ad["coastline"] = False
    ad.setdefault("coastline", USER_CONFIG.get("coastline", True))
    if ad.get("bnd"):
        ad["bnd"] = True
    ad.setdefault("bnd", USER_CONFIG.get("bnd", False))
    ad.setdefault("bnd_color", USER_CONFIG.get("bnd_color", "k"))
    ad.setdefault("bnd_lw", USER_CONFIG.get("bnd_lw", 0.6))
    ad.setdefault("mpi", USER_CONFIG.get("mpi", True))
    ad.setdefault("tmean", USER_CONFIG.get("tmean", "none"))
    ad.setdefault("mpi_show_assignment", USER_CONFIG.get("mpi_show_assignment", True))
    ad.setdefault("mpi_log_every", USER_CONFIG.get("mpi_log_every", 1))
    ad.setdefault("mpi_rank_logs", USER_CONFIG.get("mpi_rank_logs", False))
    ad.setdefault("report_stack_info", USER_CONFIG.get("report_stack_info", False))
    ad.setdefault("stack_span_hours", USER_CONFIG.get("stack_span_hours", None))
    ad.setdefault("stack_check_mode", USER_CONFIG.get("stack_check_mode", "light"))
    ad.setdefault("stack_check_all_files", USER_CONFIG.get("stack_check_all_files", False))
    ad.setdefault("stack_size_ratio_min", USER_CONFIG.get("stack_size_ratio_min", 0.70))
    ad.setdefault("stack_size_min_bytes", USER_CONFIG.get("stack_size_min_bytes", None))
    ad.setdefault("accum_dtype", USER_CONFIG.get("accum_dtype", "float32"))

    missing = [k for k in ("hgrid", "run", "outdir") if ad.get(k) is None]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}.")

    return argparse.Namespace(**ad)

def _period_key(dt, mode):
    if mode == "hourly":
        return dt.strftime("%Y-%m-%d_%H"), dt.strftime("%Y-%m-%d %H:00")
    if mode == "daily":
        return dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d")
    if mode == "monthly":
        return dt.strftime("%Y-%m"), dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d_%H%M%S"), dt.strftime("%Y-%m-%d %H:%M:%S")

def _repr_time_single_step(ctime0, span_hours=None):
    return num2date(ctime0)

def _stack_preview(stacks, limit=8):
    arr = [int(i) for i in np.asarray(stacks).ravel()]
    if len(arr) == 0:
        return "[]"
    if len(arr) <= limit:
        return "[" + ", ".join(str(i) for i in arr) + "]"
    head = ", ".join(str(i) for i in arr[:4])
    tail = ", ".join(str(i) for i in arr[-3:])
    return f"[{head}, ..., {tail}]"

def main():
    args = _merge_config(parse_args())

    mpi_enabled = (not getattr(args, "no_mpi", False)) and (_SIZE > 1) and bool(getattr(args, "mpi", True))
    rank = _RANK if mpi_enabled else 0
    size = _SIZE if mpi_enabled else 1
    skip_existing = bool(getattr(args, "skip_existing", False))
    rank_start = time.time()
    rank_log_fh = None

    outdir = Path(args.outdir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        print(
            f"[rank 0] exp={getattr(args,'exp',None)}\n"
            f"[rank 0] hgrid={args.hgrid}\n"
            f"[rank 0] run={args.run}\n"
            f"[rank 0] outdir={args.outdir}\n"
            f"[rank 0] skip_existing={skip_existing}\n"
            f"[rank 0] time_origin_mode={'param.nml' if args.apply_param_start_time else 'refdate'}\n"
            f"[rank 0] tmean={args.tmean} (agg={args.agg})\n"
            f"[rank 0] accum_dtype={args.accum_dtype}",
            flush=True
        )
    if mpi_enabled:
        _COMM.Barrier()

    if mpi_enabled and bool(getattr(args, "mpi_rank_logs", False)):
        rank_log_path = outdir / f"mpi_rank_{rank:04d}.log"
        rank_log_fh = open(rank_log_path, "w", encoding="utf-8")
        rank_log_fh.write(f"[rank {rank}/{size}] pid={os.getpid()} started\n")
        rank_log_fh.flush()

    def rlog(msg):
        line = f"[rank {rank}/{size}] {msg}"
        print(line, flush=True)
        if rank_log_fh is not None:
            rank_log_fh.write(line + "\n")
            rank_log_fh.flush()

    # ----- grid -----
    gd = read_schism_hgrid(args.hgrid)

    # Choose central_longitude to center the domain (no split across map edge),
    # then shift grid longitudes into that axes' CRS coordinates (-180..180].
    lon0_360 = _mod360(gd.x)
    central_lon = _choose_central_longitude(lon0_360)
    gd.x = _wrap_to_axis(lon0_360, central_lon)  # mutate once for consistent plotting

    proj = get_projection(args.proj, central_longitude=central_lon)
    data_crs = ccrs.PlateCarree(central_longitude=central_lon)

    if rank == 0:
        print(f"[rank 0] central_longitude chosen = {central_lon:.2f}°", flush=True)

    if args.vars:
        vars_to_plot = args.vars
    else:
        vars_to_plot = LAYOUT_TO_VARS.get(args.layout, ["ssh", "uv"])

    if args.layout == "SINGLE" and len(vars_to_plot) != 1:
        if rank == 0:
            print("When --layout=SINGLE, pass exactly one variable via --vars", file=sys.stderr)
        sys.exit(2)

    cands = _select_candidate_stacks(args.run, start=args.start, end=args.end)
    if len(cands) == 0:
        if rank == 0:
            print(f"[rank 0] No stack files found in {args.run} for range start={args.start}, end={args.end}.",
                  file=sys.stderr, flush=True)
        sys.exit(2)

    stacks, skipped = _screen_stacks(
        args.run,
        cands,
        vars_to_plot,
        mode=args.stack_check_mode,
        check_all_files=bool(args.stack_check_all_files),
        ratio_min=float(args.stack_size_ratio_min),
        abs_min_bytes=args.stack_size_min_bytes,
    )
    if rank == 0:
        print(f"[rank 0] stack screening mode={args.stack_check_mode}: "
              f"candidates={len(cands)}, valid={len(stacks)}, skipped={len(skipped)}", flush=True)
        for st in sorted(skipped)[:20]:
            print(f"[rank 0]   skip stack {st}: {skipped[st]}", flush=True)
        if len(skipped) > 20:
            print(f"[rank 0]   ... {len(skipped) - 20} more skipped stacks", flush=True)
    if len(stacks) == 0:
        if rank == 0:
            print("[rank 0] No valid stacks left after screening.", file=sys.stderr, flush=True)
        sys.exit(2)

    time_origin_dnum, time_origin_desc = _get_time_origin_datenum(args)
    if rank == 0:
        print(f"[rank 0] time origin: {time_origin_desc}", flush=True)

    local_stacks = stacks[rank::size]
    if rank == 0 and size > 1:
        print(f"MPI: world size={size}; total stacks={len(stacks)}; rank0 handles {len(local_stacks)}", flush=True)
    if mpi_enabled and bool(getattr(args, "mpi_show_assignment", True)):
        assign = _COMM.gather([int(i) for i in local_stacks], root=0)
        if rank == 0:
            print("[rank 0] MPI rank assignment summary:", flush=True)
            for rr, arr in enumerate(assign):
                first = arr[0] if len(arr) > 0 else None
                last = arr[-1] if len(arr) > 0 else None
                print(
                    f"[rank 0]   rank {rr}: n={len(arr)}, first={first}, last={last}, stacks={_stack_preview(arr)}",
                    flush=True,
                )
    rlog(f"starting processing: local_stacks={len(local_stacks)}, preview={_stack_preview(local_stacks)}")

    do_tmean = args.tmean in ("hourly", "daily", "monthly")
    accum_dtype = np.float32 if str(args.accum_dtype) == "float32" else np.float64
    log_every = max(1, int(getattr(args, "mpi_log_every", 1)))

    sums   = {v: {} for v in vars_to_plot}
    counts = {v: {} for v in vars_to_plot}
    labels = {}
    local_keys_set = set()
    n_nodes = None

    total_local = len(local_stacks)
    done_count = 0

    for istack in local_stacks:
        cadence_done = False
        time_rep = None

        if not do_tmean:
            stack_fields = {}
            stack_masks = {}
            ctime_ref = None

            for var in vars_to_plot:
                ctime, arr2d, mask2d = read_field(args.run, var, istack, args.layer, time_origin_dnum)
                ctime = np.asarray(ctime, dtype=np.float64)

                if not cadence_done:
                    n_steps = len(ctime)
                    if n_steps >= 2:
                        dt_sec = np.diff(np.asarray(ctime, dtype=np.float64)) * 86400.0
                        med_dt_h = float(np.nanmedian(dt_sec) / 3600.0)
                        span_h   = float((ctime[-1] - ctime[0]) * 24.0)
                        first_dt = num2date(ctime[0]); last_dt = num2date(ctime[-1])
                        if args.report_stack_info:
                            print(f"[rank {rank}/{size}] stack {istack}: steps={n_steps}, "
                                  f"median_dt={med_dt_h:.3f} h, span={span_h:.3f} h, "
                                  f"first={first_dt}, last={last_dt}", flush=True)
                        time_rep = num2date(np.nanmean(ctime))
                    else:
                        time_rep = _repr_time_single_step(ctime[0], span_hours=args.stack_span_hours)
                        if args.report_stack_info:
                            print(f"[rank {rank}/{size}] stack {istack}: single-step; "
                                  f"time={time_rep}", flush=True)
                    cadence_done = True

                if ctime_ref is None:
                    ctime_ref = ctime
                stack_fields[var] = np.asarray(arr2d, dtype=np.float32)
                stack_masks[var] = (np.asarray(mask2d) if mask2d is not None else None)
                if n_nodes is None:
                    n_nodes = arr2d.shape[1]

                del arr2d

            want_cbar_each = (args.cbars == "each")
            want_cbar_none = (args.cbars == "none")

            if str(args.agg) == "file-mean":
                plot_fields = {}
                for var in vars_to_plot:
                    reduced = np.nanmean(stack_fields[var], axis=0).astype(np.float32, copy=False)
                    mask2d = stack_masks.get(var)
                    if mask2d is not None and mask2d.ndim == 2:
                        always_dry = (mask2d == 1).all(axis=0)
                        reduced = np.asarray(reduced)
                        reduced[always_dry] = np.nan
                    plot_fields[var] = reduced

                suptitle = time_rep.strftime("%Y-%m-%d")
                fname = (
                    (f"{args.prefix}_" if args.prefix else "")
                    + f"stack{int(istack):04d}_{suptitle.replace(' ', '_').replace(':', '')}_filemean.png"
                )
                fpath = outdir / fname
                if skip_existing and fpath.exists():
                    rlog(f"skip existing {fname}")
                else:
                    fig, axes = make_axes(len(vars_to_plot), proj, args.layout)
                    for ax, var in zip(axes, vars_to_plot):
                        cfg = VAR_DEFAULTS[var]
                        cmap = _get_cmap(cfg["cmap"]); clim = cfg["clim"]
                        format_geoaxes(
                            ax, gd, args.extent, title_txt=cfg["title"], data_crs=data_crs,
                            show_coastline=args.coastline, show_bnd=args.bnd,
                            bnd_color=args.bnd_color, bnd_lw=args.bnd_lw
                        )

                        plt.sca(ax)
                        m = gd.plot(fmt=1, value=plot_fields[var], cmap=cmap, clim=clim, cb=False, ax=ax)

                        if not want_cbar_none and want_cbar_each:
                            plt.colorbar(
                                m, ax=ax,
                                orientation="vertical" if len(vars_to_plot) in (1, 2) else "horizontal",
                                fraction=0.05, pad=0.10
                            )

                    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
                    if not fig.get_constrained_layout():
                        fig.tight_layout()
                    fig.savefig(fpath, dpi=args.dpi)
                    plt.close(fig)
                    rlog(f"wrote {fname}")
                del plot_fields

            else:
                n_steps = int(len(ctime_ref)) if ctime_ref is not None else 0
                for it in range(n_steps):
                    dt = num2date(ctime_ref[it])
                    suptitle = dt.strftime("%Y-%m-%d %H:%M:%S")
                    safe_t = suptitle.replace(" ", "_").replace(":", "")
                    fname = (f"{args.prefix}_" if args.prefix else "") + f"{safe_t}.png"
                    fpath = outdir / fname
                    if skip_existing and fpath.exists():
                        if (it == 0) or ((it + 1) % 10 == 0) or ((it + 1) == n_steps):
                            rlog(f"stack {istack}: skip existing step {it + 1}/{n_steps}")
                        continue

                    plot_fields = {}
                    for var in vars_to_plot:
                        x = np.asarray(stack_fields[var][it, :], dtype=np.float32)
                        mask2d = stack_masks.get(var)
                        if mask2d is not None and mask2d.ndim == 2:
                            dry = (mask2d[it, :] == 1)
                            x = x.copy()
                            x[dry] = np.nan
                        plot_fields[var] = x

                    fig, axes = make_axes(len(vars_to_plot), proj, args.layout)
                    for ax, var in zip(axes, vars_to_plot):
                        cfg = VAR_DEFAULTS[var]
                        cmap = _get_cmap(cfg["cmap"]); clim = cfg["clim"]
                        format_geoaxes(
                            ax, gd, args.extent, title_txt=cfg["title"], data_crs=data_crs,
                            show_coastline=args.coastline, show_bnd=args.bnd,
                            bnd_color=args.bnd_color, bnd_lw=args.bnd_lw
                        )

                        plt.sca(ax)
                        m = gd.plot(fmt=1, value=plot_fields[var], cmap=cmap, clim=clim, cb=False, ax=ax)

                        if not want_cbar_none and want_cbar_each:
                            plt.colorbar(
                                m, ax=ax,
                                orientation="vertical" if len(vars_to_plot) in (1, 2) else "horizontal",
                                fraction=0.05, pad=0.10
                            )

                    fig.suptitle(suptitle, fontsize=11, fontweight="bold")
                    if not fig.get_constrained_layout():
                        fig.tight_layout()
                    fig.savefig(fpath, dpi=args.dpi)
                    plt.close(fig)

                    if (it == 0) or ((it + 1) % 10 == 0) or ((it + 1) == n_steps):
                        rlog(f"stack {istack}: wrote step {it + 1}/{n_steps}")

                    del plot_fields

            for var in list(stack_fields.keys()):
                del stack_fields[var]
            for var in list(stack_masks.keys()):
                del stack_masks[var]
            del stack_fields, stack_masks, ctime_ref
            gc.collect()

        else:
            first_var_seen = True
            for var in vars_to_plot:
                ctime, arr2d, mask2d = read_field(args.run, var, istack, args.layer, time_origin_dnum)

                if first_var_seen:
                    n_steps = len(ctime)
                    if n_steps >= 2:
                        dt_sec = np.diff(np.asarray(ctime, dtype=np.float64)) * 86400.0
                        med_dt_h = float(np.nanmedian(dt_sec) / 3600.0)
                        span_h   = float((ctime[-1] - ctime[0]) * 24.0)
                        first_dt = num2date(ctime[0]); last_dt = num2date(ctime[-1])
                        if args.report_stack_info:
                            print(f"[rank {rank}/{size}] stack {istack}: steps={n_steps}, "
                                  f"median_dt={med_dt_h:.3f} h, span={span_h:.3f} h, "
                                  f"first={first_dt}, last={last_dt}", flush=True)
                    else:
                        first_dt = num2date(ctime[0])
                        if args.report_stack_info:
                            print(f"[rank {rank}/{size}] stack {istack}: single-step; time={first_dt}", flush=True)
                    first_var_seen = False

                if mask2d is not None and mask2d.ndim == 2:
                    dry = (mask2d == 1)
                    arr2d = np.asarray(arr2d)
                    arr2d[dry] = np.nan
                    del dry, mask2d

                for it in range(arr2d.shape[0]):
                    dt = num2date(ctime[it])
                    key, label = _period_key(dt, args.tmean)
                    labels[key] = label
                    local_keys_set.add(key)

                    x = arr2d[it, :]
                    if key not in sums[var]:
                        if n_nodes is None:
                            n_nodes = x.shape[0]
                        sums[var][key]   = np.zeros((n_nodes,), dtype=accum_dtype)
                        counts[var][key] = np.zeros((n_nodes,), dtype=accum_dtype)

                    valid = ~np.isnan(x)
                    sums[var][key][valid]   += x[valid].astype(accum_dtype, copy=False)
                    counts[var][key][valid] += 1.0

                del arr2d
                gc.collect()

        done_count += 1
        if (done_count % log_every == 0) or (done_count == total_local):
            elapsed = time.time() - rank_start
            rlog(f"done {done_count}/{total_local} stacks; latest={istack}; elapsed={elapsed:.1f}s")

    rank_elapsed = time.time() - rank_start
    if mpi_enabled:
        rank_stats = _COMM.gather((rank, done_count, total_local, rank_elapsed), root=0)
    else:
        rank_stats = [(rank, done_count, total_local, rank_elapsed)]

    if not do_tmean:
        if mpi_enabled:
            _COMM.Barrier()
        if rank == 0:
            print("[rank 0] MPI rank timing summary:", flush=True)
            for rr, dc, tl, dt in sorted(rank_stats, key=lambda x: x[0]):
                print(f"[rank 0]   rank {rr}: processed={dc}/{tl}, elapsed={dt:.1f}s", flush=True)
            print("---------DONE---------")
        if rank_log_fh is not None:
            rank_log_fh.write(f"[rank {rank}/{size}] finished in {rank_elapsed:.1f}s\n")
            rank_log_fh.close()
        return

    # ---- Temporal mean across stacks (MPI reduce) ----
    if mpi_enabled:
        local_keys = sorted(local_keys_set)
        all_keys_lists = _COMM.allgather(local_keys)
        global_keys = sorted(set(k for sub in all_keys_lists for k in sub))
    else:
        global_keys = sorted(local_keys_set)

    for ikey, key in enumerate(global_keys):
        mean_fields = {}
        for var in vars_to_plot:
            if key in sums[var]:
                local_sum = sums[var][key]
                local_cnt = counts[var][key]
            else:
                if n_nodes is None:
                    local_sum = np.array([], dtype=accum_dtype)
                    local_cnt = np.array([], dtype=accum_dtype)
                else:
                    local_sum = np.zeros((n_nodes,), dtype=accum_dtype)
                    local_cnt = np.zeros((n_nodes,), dtype=accum_dtype)

            if mpi_enabled:
                max_len = _COMM.allreduce(local_sum.shape[0], op=MPI.MAX)
                if local_sum.shape[0] != max_len:
                    zsum = np.zeros((max_len,), dtype=accum_dtype); zsum[:local_sum.shape[0]] = local_sum
                    zcnt = np.zeros((max_len,), dtype=accum_dtype); zcnt[:local_cnt.shape[0]] = local_cnt
                    local_sum, local_cnt = zsum, zcnt

                global_sum = np.zeros_like(local_sum)
                global_cnt = np.zeros_like(local_cnt)
                _COMM.Allreduce(local_sum, global_sum, op=MPI.SUM)
                _COMM.Allreduce(local_cnt, global_cnt, op=MPI.SUM)
            else:
                global_sum = local_sum
                global_cnt = local_cnt

            with np.errstate(invalid="ignore", divide="ignore"):
                mean_arr = np.where(global_cnt > 0, global_sum / global_cnt, np.nan).astype(np.float32, copy=False)

            mean_fields[var] = mean_arr

        # In MPI, distribute temporal-bucket figure writing across ranks.
        write_this_rank = ((ikey % size) == rank) if mpi_enabled else (rank == 0)

        if write_this_rank:
            label = global_keys and labels.get(key, key) or key
            safe_key = key.replace(":", "").replace(" ", "_")
            fname = (f"{args.prefix}_" if args.prefix else "") + safe_key + f"_{args.tmean}.png"
            fpath = outdir / fname
            if skip_existing and fpath.exists():
                print(f"[rank {rank}] skip existing {fname}", flush=True)
                for var in list(mean_fields.keys()):
                    del mean_fields[var]
                del mean_fields
                gc.collect()
                continue

            fig, axes = make_axes(len(vars_to_plot), proj, args.layout)
            want_cbar_each = (args.cbars == "each")
            want_cbar_none = (args.cbars == "none")
            for ax, var in zip(axes, vars_to_plot):
                cfg = VAR_DEFAULTS[var]
                cmap = _get_cmap(cfg["cmap"]); clim = cfg["clim"]
                format_geoaxes(
                    ax, gd, None, title_txt=cfg["title"], data_crs=data_crs,
                    show_coastline=args.coastline, show_bnd=args.bnd,
                    bnd_color=args.bnd_color, bnd_lw=args.bnd_lw
                )

                plt.sca(ax)  # ensure drawing on THIS axes
                m = gd.plot(fmt=1, value=mean_fields[var], cmap=cmap, clim=clim, cb=False, ax=ax)

                if not want_cbar_none and want_cbar_each:
                    plt.colorbar(
                        m, ax=ax,
                        orientation="vertical" if len(vars_to_plot) in (1, 2) else "horizontal",
                        fraction=0.05, pad=0.10
                    )

            fig.suptitle(label + "  (mean)", fontsize=11, fontweight="bold")
            if not fig.get_constrained_layout():
                fig.tight_layout()
            fig.savefig(fpath, dpi=args.dpi)
            plt.close(fig)
            print(f"[rank {rank}] wrote {fname}", flush=True)

        for var in list(mean_fields.keys()):
            del mean_fields[var]
        del mean_fields
        gc.collect()

    if mpi_enabled:
        _COMM.Barrier()
    if rank == 0:
        print("[rank 0] MPI rank timing summary:", flush=True)
        for rr, dc, tl, dt in sorted(rank_stats, key=lambda x: x[0]):
            print(f"[rank 0]   rank {rr}: processed={dc}/{tl}, elapsed={dt:.1f}s", flush=True)
        print("---------DONE---------")
    if rank_log_fh is not None:
        rank_log_fh.write(f"[rank {rank}/{size}] finished in {rank_elapsed:.1f}s\n")
        rank_log_fh.close()

if __name__ == "__main__":
    main()
