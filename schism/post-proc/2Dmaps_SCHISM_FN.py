#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCHISM 2D Map Maker — integrated, optimized, and MPI-enabled
with temporal aggregation: --tmean {none,hourly,daily,monthly}
"""

# ---- headless & cluster guards ----
import os
os.environ.setdefault("MPLBACKEND", "Agg")
if "frontera" in (os.environ.get("HOSTNAME", "").lower(), os.environ.get("TACC_SYSTEM", "").lower()):
    os.environ["TACC_SYSTEM"] = "headless"

# ---- std libs ----
import sys
import argparse
from pathlib import Path
import numpy as np
import gc

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
from pylib import read_schism_hgrid, ReadNC, datenum, num2date

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

# =====================
# Config
# =====================
USER_CONFIG = {
    "enable": True,

    # --- Experiment-driven paths ---
    "exp": "RUN03b",                                # e.g., RUN01a/RUN01b/...
    "base_run_dir": "/storage/coda1/p-ed70/0/kpark350/Projects/OTEC_GUAM/run/", # parent dir for runs
    "base_outdir": "./images/RUN03b",                      # parent dir for images

    # If any are None, they'll be derived from exp:
    "hgrid": None,   # -> {base_run_dir}/{exp}/hgrid.gr3
    "run": None,     # -> {base_run_dir}/{exp}/outputs
    "outdir": None,  # -> {base_outdir}/{exp}

    # Range & time
    "start": 3000,
    "end": 4140,
    "refdate": "2011-04-02",

    # Layer & intra-file aggregation
    "layer": -1,              # -1 = surface
    "agg": "file-mean",       # or "inst"

    # Layout & variables
    "layout": "TS",          # EC, TS, ALL, SINGLE
    "vars": None,

    # Plot
    "prefix": "",
    "dpi": 500,
    "cbars": "each",          # each, shared, none
    "proj": "PlateCarree",
    "extent": None,           # [xmin xmax ymin ymax]

    # MPI & temporal mean
    "mpi": True,
    "tmean": "hourly",          # none | hourly | daily | monthly

    # Reporting
    "report_stack_info": False,
    "stack_span_hours": None,

    # Temporal-mean accumulator dtype
    "accum_dtype": "float32", # float32 | float64
}

# -------------------------
# Helpers
# -------------------------
VAR_DEFAULTS = {
    "ssh":  {"cmap": "cmocean.balance", "clim": (-1.5, 1.5), "title": "Sea Surface Height (m)"},
    "uv":   {"cmap": "cmocean.speed",   "clim": (0.0, 1.0),   "title": "Surface Current Speed (m s$^{-1}$)"},
    "temp": {"cmap": "cmocean.thermal", "clim": (19.0, 32.0),  "title": "Temperature (°C)"},
    "salt": {"cmap": "cmocean.haline",  "clim": (0.0, 36.0),  "title": "Salinity (psu)"},
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
    p.add_argument("--layer", type=int)
    p.add_argument("--agg", choices=["file-mean", "inst"])
    p.add_argument("--layout", choices=["EC", "TS", "ALL", "SINGLE"])
    p.add_argument("--vars", nargs="+", choices=list(VAR_DEFAULTS.keys()))
    p.add_argument("--prefix")
    p.add_argument("--dpi", type=int)
    p.add_argument("--cbars", choices=["each", "shared", "none"])
    p.add_argument("--proj")
    p.add_argument("--extent", nargs=4, type=float)
    # Temporal mean
    p.add_argument("--tmean", choices=["none", "hourly", "daily", "monthly"])
    # Reporting
    p.add_argument("--report-stack-info", action="store_true")
    p.add_argument("--stack-span-hours", type=float)
    # Accumulator dtype
    p.add_argument("--accum-dtype", choices=["float32", "float64"])
    # Toggles
    p.add_argument("--no-config", action="store_true")
    p.add_argument("--no-mpi", action="store_true")
    return p.parse_args()

def get_projection(name: str, central_longitude: float = 0.0):
    if name == "PlateCarree":
        return ccrs.PlateCarree(central_longitude=central_longitude)
    if name == "Mercator":
        return ccrs.Mercator()
    if name == "LambertConformal":
        return ccrs.LambertConformal()
    return ccrs.PlateCarree(central_longitude=central_longitude)

def read_time_and_mask_nc(nc, refdate):
    t = nc.variables["time"][:]  # seconds since ref
    ctime = t / 86400.0 + datenum(refdate)
    mvar = nc.variables.get("dryFlagNode")
    mask = _as_f32_with_nan(mvar[:]) if mvar is not None else None
    return ctime, (np.asarray(mask) if mask is not None else None)

# -------- Memory-lean readers (single-layer) --------
def read_field(run_dir, var, istack, layer_idx, refdate, out2d_for_mask=True):
    if var == "ssh":
        p = f"{run_dir}/out2d_{istack}.nc"
        with Dataset(p, "r") as nc:
            ctime, mask = read_time_and_mask_nc(nc, refdate)
            elev = _as_f32_with_nan(nc.variables["elevation"][:])  # [time,node]
        return ctime, elev, mask

    if var == "uv":
        up = f"{run_dir}/horizontalVelX_{istack}.nc"
        vp = f"{run_dir}/horizontalVelY_{istack}.nc"
        with Dataset(up, "r") as ncu, Dataset(vp, "r") as ncv:
            ctime, mask = read_time_and_mask_nc(ncu, refdate)
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
            ctime, mask = read_time_and_mask_nc(nc, refdate)
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

def format_geoaxes(ax, gd, extent, title_txt=None, data_crs=None):
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
    ad.setdefault("layer", -1)
    ad.setdefault("agg", "file-mean")
    ad.setdefault("layout", "EC")
    ad.setdefault("dpi", 200)
    ad.setdefault("cbars", "each")
    ad.setdefault("proj", "PlateCarree")
    ad.setdefault("mpi", USER_CONFIG.get("mpi", True))
    ad.setdefault("tmean", USER_CONFIG.get("tmean", "none"))
    ad.setdefault("report_stack_info", USER_CONFIG.get("report_stack_info", False))
    ad.setdefault("stack_span_hours", USER_CONFIG.get("stack_span_hours", None))
    ad.setdefault("accum_dtype", USER_CONFIG.get("accum_dtype", "float32"))

    missing = [k for k in ("hgrid", "run", "start", "end", "outdir") if ad.get(k) is None]
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

def main():
    args = _merge_config(parse_args())

    mpi_enabled = (not getattr(args, "no_mpi", False)) and (_SIZE > 1) and bool(getattr(args, "mpi", True))
    rank = _RANK if mpi_enabled else 0
    size = _SIZE if mpi_enabled else 1

    outdir = Path(args.outdir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        print(
            f"[rank 0] exp={getattr(args,'exp',None)}\n"
            f"[rank 0] hgrid={args.hgrid}\n"
            f"[rank 0] run={args.run}\n"
            f"[rank 0] outdir={args.outdir}\n"
            f"[rank 0] tmean={args.tmean} (agg={args.agg})\n"
            f"[rank 0] accum_dtype={args.accum_dtype}",
            flush=True
        )
    if mpi_enabled:
        _COMM.Barrier()

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

    stacks = list(range(args.start, args.end + 1))
    local_stacks = stacks[rank::size]
    if rank == 0 and size > 1:
        print(f"MPI: world size={size}; total stacks={len(stacks)}; rank0 handles {len(local_stacks)}", flush=True)

    do_tmean = args.tmean in ("hourly", "daily", "monthly")
    accum_dtype = np.float32 if str(args.accum_dtype) == "float32" else np.float64

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
            reduced_fields = {}

            for var in vars_to_plot:
                ctime, arr2d, mask2d = read_field(args.run, var, istack, args.layer, args.refdate)

                if not cadence_done:
                    n_steps = len(ctime)
                    if n_steps >= 2:
                        dt_sec = np.diff(ctime) * 86400.0
                        med_dt_h = float(np.median(dt_sec) / 3600.0)
                        span_h   = float((ctime[-1] - ctime[0]) * 24.0)
                        first_dt = num2date(ctime[0]); last_dt = num2date(ctime[-1])
                        if args.report_stack_info:
                            print(f"[rank {rank}/{size}] stack {istack}: steps={n_steps}, "
                                  f"median_dt={med_dt_h:.3f} h, span={span_h:.3f} h, "
                                  f"first={first_dt}, last={last_dt}", flush=True)
                        time_rep = num2date(np.mean(ctime))
                    else:
                        time_rep = _repr_time_single_step(ctime[0], span_hours=args.stack_span_hours)
                        if args.report_stack_info:
                            print(f"[rank {rank}/{size}] stack {istack}: single-step; "
                                  f"time={time_rep}", flush=True)
                    cadence_done = True

                reduced = reduce_time(arr2d, args.agg)

                if mask2d is not None and mask2d.ndim == 2:
                    always_dry = (mask2d == 1).all(axis=0)
                    reduced = np.asarray(reduced)
                    reduced[always_dry] = np.nan

                reduced_fields[var] = reduced.astype(np.float32, copy=False)
                if n_nodes is None:
                    n_nodes = reduced.shape[0]

                del arr2d
                if mask2d is not None:
                    del mask2d

            fig, axes = make_axes(len(vars_to_plot), proj, args.layout)
            want_cbar_each = (args.cbars == "each"); want_cbar_none = (args.cbars == "none")
            for ax, var in zip(axes, vars_to_plot):
                cfg = VAR_DEFAULTS[var]
                cmap = _get_cmap(cfg["cmap"]); clim = cfg["clim"]
                format_geoaxes(ax, gd, args.extent, title_txt=cfg["title"], data_crs=data_crs)

                plt.sca(ax)  # ensure drawing on THIS axes
                m = gd.plot(fmt=1, value=reduced_fields[var], cmap=cmap, clim=clim, cb=False, ax=ax)

                if not want_cbar_none and want_cbar_each:
                    plt.colorbar(
                        m, ax=ax,
                        orientation="vertical" if len(vars_to_plot) in (1, 2) else "horizontal",
                        fraction=0.05, pad=0.10
                    )

            suptitle = time_rep.strftime("%Y-%m-%d %H:%M:%S") if args.agg == "inst" else time_rep.strftime("%Y-%m-%d")
            fig.suptitle(suptitle, fontsize=11, fontweight="bold")
            if not fig.get_constrained_layout():
                fig.tight_layout()
            fname = (f"{args.prefix}_" if args.prefix else "") + suptitle.replace(" ", "_").replace(":", "") + ".png"
            fig.savefig(outdir / fname, dpi=args.dpi)
            plt.close(fig)

            for var in list(reduced_fields.keys()):
                del reduced_fields[var]
            del reduced_fields
            gc.collect()

        else:
            first_var_seen = True
            for var in vars_to_plot:
                ctime, arr2d, mask2d = read_field(args.run, var, istack, args.layer, args.refdate)

                if first_var_seen:
                    n_steps = len(ctime)
                    if n_steps >= 2:
                        dt_sec = np.diff(ctime) * 86400.0
                        med_dt_h = float(np.median(dt_sec) / 3600.0)
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
        print(f"[rank {rank}/{size}] done {done_count}/{total_local} → stack {istack}", flush=True)

    if not do_tmean:
        if mpi_enabled:
            _COMM.Barrier()
        if rank == 0:
            print("---------DONE---------")
        return

    # ---- Temporal mean across stacks (MPI reduce) ----
    if mpi_enabled:
        local_keys = sorted(local_keys_set)
        all_keys_lists = _COMM.allgather(local_keys)
        global_keys = sorted(set(k for sub in all_keys_lists for k in sub))
    else:
        global_keys = sorted(local_keys_set)

    for key in global_keys:
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

        if rank == 0:
            fig, axes = make_axes(len(vars_to_plot), proj, args.layout)
            want_cbar_each = (args.cbars == "each")
            want_cbar_none = (args.cbars == "none")
            label = global_keys and labels.get(key, key) or key
            for ax, var in zip(axes, vars_to_plot):
                cfg = VAR_DEFAULTS[var]
                cmap = _get_cmap(cfg["cmap"]); clim = cfg["clim"]
                format_geoaxes(ax, gd, None, title_txt=cfg["title"], data_crs=data_crs)

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
            safe_key = key.replace(":", "").replace(" ", "_")
            fname = (f"{args.prefix}_" if args.prefix else "") + safe_key + f"_{args.tmean}.png"
            fig.savefig(outdir / fname, dpi=args.dpi)
            plt.close(fig)
            print(f"[rank 0] wrote {fname}", flush=True)

        for var in list(mean_fields.keys()):
            del mean_fields[var]
        del mean_fields
        gc.collect()

    if mpi_enabled:
        _COMM.Barrier()
    if rank == 0:
        print("---------DONE---------")

if __name__ == "__main__":
    main()
