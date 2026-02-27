#!/usr/bin/env python3
"""
WL-only SCHISM vs JODC tide comparison.

This script is a streamlined replacement for WL workflows in comp_schism_th.py.
It keeps the original script untouched and focuses only on:
  - variable: WL
  - observation file: jodc_tide_all.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylib import datenum, loadz, num2date, read, read_schism_bpfile

try:
    from scipy import signal
except Exception:  # pragma: no cover - optional dependency
    signal = None

MPI = None
COMM = None
RANK = 0
SIZE = 1


def _env_int(names: Sequence[str], default: int = 0) -> int:
    for name in names:
        val = os.environ.get(name)
        if val is None:
            continue
        try:
            return int(val)
        except Exception:
            continue
    return default


def _looks_like_mpi_launch() -> bool:
    size = _env_int(
        [
            "OMPI_COMM_WORLD_SIZE",
            "PMI_SIZE",
            "PMIX_SIZE",
            "MPI_LOCALNRANKS",
            "SLURM_NTASKS",
        ],
        default=0,
    )
    return size > 1


USE_MPI = (
    "--mpi" in sys.argv
    or os.environ.get("ENABLE_MPI", "0") == "1"
    or _looks_like_mpi_launch()
)
if "--no-mpi" in sys.argv:
    USE_MPI = False
if "--mpi" in sys.argv:
    sys.argv.remove("--mpi")
if "--no-mpi" in sys.argv:
    sys.argv.remove("--no-mpi")

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


SCRIPT_DIR = Path(__file__).resolve().parent
OBS_DEFAULT = SCRIPT_DIR / "npz" / "jodc_tide_all.npz"


DEFAULT_CONFIG: Dict[str, Any] = {
    "runs": ["/scratch2/08924/kmpark/post-proc/npz/RUN01e_JODC.npz", "/scratch2/08924/kmpark/post-proc/npz/RUN01f_JODC.npz", "/scratch2/08924/kmpark/post-proc/npz/RUN02a_JODC.npz", "/scratch2/08924/kmpark/post-proc/npz/RUN02b_JODC.npz"],
    "tags": ["RUN01e", "RUN01f", "RUN02a", "RUN02b"],
    "bpfile": "/scratch2/08924/kmpark/post-proc/station_jodc.bp",
    "outdir": "/scratch2/08924/kmpark/post-proc/ESIMAGES/CompJODC_01ef02ab",
    "obs_path": "/scratch2/08924/kmpark/post-proc/npz/jodc_tide_all.npz",
    "start": "2017-04-14 00:00:00",
    "end": "2017-04-30 00:00:00",
    "model_start": None,
    "model_time_offset_days": [0.0],
    "npz_time_mode": "absolute",  # absolute | relative | auto
    "apply_offset_to_npz": False,  # legacy fallback when npz_time_mode=absolute
    "resample_obs": "h",
    "resample_model": "h",
    "demean": True,
    "plot_ylim": [-1.5, 1.5],
    "line_width": 1.8,
    "save_plots": True,
    "filter_obs": False,
    "filter_model": False,
    "cutoff_period_hours": 34.0,
    "butterworth_order": 4,
    "station_list": None,
    "progress_every": 1,
    "grid": None,  # optional hgrid.gr3 for map boundary
    "map_zoom": 0.1,  # half-width in degree around active station
}


def log(msg: str, rank0_only: bool = False) -> None:
    if rank0_only and RANK != 0:
        return
    prefix = f"[rank {RANK}/{SIZE}] " if SIZE > 1 else ""
    print(prefix + msg, flush=True)


def _report_station_assignment(tag: str, total_count: int, local_indices: Sequence[int]) -> None:
    nloc = len(local_indices)
    if nloc > 0:
        first_idx = int(local_indices[0])
        last_idx = int(local_indices[-1])
    else:
        first_idx = -1
        last_idx = -1
    log(
        f"{tag} assignment: local={nloc}/{total_count}, "
        f"index_range=[{first_idx},{last_idx}], stride={SIZE}",
        rank0_only=False,
    )
    if MPI:
        counts = COMM.gather(nloc, root=0)
        if RANK == 0:
            summary = ", ".join([f"r{i}:{c}" for i, c in enumerate(counts)])
            log(f"{tag} distribution by rank -> {summary}", rank0_only=True)
        COMM.Barrier()


def _resolve_path(path_like: str) -> Path:
    path = Path(os.path.expanduser(path_like))
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def _parse_time_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(datenum(str(value)))


def _expand_scalar_or_list(values: Sequence[Any], n: int) -> List[Any]:
    seq = list(values)
    if len(seq) == 1 and n > 1:
        return seq * n
    if len(seq) != n:
        raise ValueError(f"Expected {n} values but received {len(seq)}")
    return seq


def _resample(times: np.ndarray, values: np.ndarray, freq: Optional[str]) -> pd.Series:
    if len(times) == 0:
        return pd.Series(dtype=float)
    stamps = [num2date(float(t)).strftime("%Y-%m-%d %H:%M:%S") for t in times]
    s = pd.Series(values.astype(float), index=pd.to_datetime(stamps, utc=True)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if freq:
        f = str(freq).strip().lower()
        s = s.resample(f).mean()
    return s.dropna()


def _apply_lowpass(
    values: np.ndarray,
    times: np.ndarray,
    cutoff_hours: float,
    order: int,
) -> np.ndarray:
    if signal is None:
        return values
    if len(values) < max(order * 3, 8):
        return values
    dt_hours = np.median(np.diff(times)) * 24.0
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        return values
    cutoff_freq = 1.0 / float(cutoff_hours)
    nyquist = 0.5 / dt_hours
    if nyquist <= 0:
        return values
    wn = cutoff_freq / nyquist
    if not np.isfinite(wn) or wn <= 0:
        return values
    wn = min(0.999, float(wn))
    if wn <= 0:
        return values
    try:
        b, a = signal.butter(int(order), wn, btype="low")
        return signal.filtfilt(b, a, values)
    except Exception:
        return values


def _compute_metrics(obs: np.ndarray, mod: np.ndarray) -> Dict[str, float]:
    if len(obs) == 0 or len(mod) == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "rmse": np.nan,
            "corr": np.nan,
            "obs_std": np.nan,
            "mod_std": np.nan,
            "nrmse_std": np.nan,
            "wss": np.nan,
            "crmsd": np.nan,
        }
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(mod)
    obs = obs[valid]
    mod = mod[valid]
    n = int(len(obs))
    if n == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "rmse": np.nan,
            "corr": np.nan,
            "obs_std": np.nan,
            "mod_std": np.nan,
            "nrmse_std": np.nan,
            "wss": np.nan,
            "crmsd": np.nan,
        }
    diff = mod - obs
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff * diff)))
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


def _load_observations(obs_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    obs = loadz(str(obs_path))
    stations = np.asarray(obs.station).astype("U")
    times_raw = np.asarray(obs.time)
    elev = np.asarray(obs.elev, dtype=float)
    if np.issubdtype(times_raw.dtype, np.datetime64):
        tstr = times_raw.astype("datetime64[s]").astype("U")
        times = np.array([float(datenum(s)) for s in tstr], dtype=float)
    else:
        times = np.asarray(times_raw, dtype=float)

    df = pd.DataFrame({"station": stations, "time": times, "elev": elev})
    df = df[np.isfinite(df["time"]) & np.isfinite(df["elev"])]
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, grp in df.groupby("station"):
        arr_t = grp["time"].to_numpy(dtype=float)
        arr_v = grp["elev"].to_numpy(dtype=float)
        order = np.argsort(arr_t)
        out[str(sid).strip()] = (arr_t[order], arr_v[order])
    return out


def _normalize_station_id(sid: str) -> str:
    return str(sid).strip()


def _build_station_lookup(obs_map: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, str]:
    lookup = {}
    for key in obs_map:
        lookup[_normalize_station_id(key)] = key
        lookup[_normalize_station_id(key).lstrip("0")] = key
    return lookup


def _npz_time_is_absolute(ds: Any, time_values: np.ndarray) -> bool:
    if hasattr(ds, "time_is_absolute"):
        try:
            return bool(int(np.asarray(ds.time_is_absolute).ravel()[0]) == 1)
        except Exception:
            pass
    if hasattr(ds, "time_units"):
        try:
            units = str(np.asarray(ds.time_units).ravel()[0]).lower()
            if "datenum" in units or "absolute" in units:
                return True
        except Exception:
            pass
    # Fallback heuristic: absolute datenums are usually very large.
    finite = np.asarray(time_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return bool(np.nanmedian(finite) > 10000.0)


def _load_model_run(
    path: Path,
    offset_days: float,
    npz_time_mode: str = "absolute",
    apply_offset_to_npz: bool = False,
) -> Dict[str, Any]:
    source_kind = "npz" if str(path).endswith(".npz") else "staout"
    mode_used = ""
    offset_applied = False
    if str(path).endswith(".npz"):
        ds = loadz(str(path))
        if not hasattr(ds, "time"):
            raise ValueError(f"{path}: missing 'time' in npz")
        if hasattr(ds, "elev"):
            elev = np.asarray(ds.elev, dtype=float)
        elif hasattr(ds, "wl"):
            elev = np.asarray(ds.wl, dtype=float)
        else:
            raise ValueError(f"{path}: missing 'elev' (or 'wl') in npz")
        t_raw = np.asarray(ds.time, dtype=float)
        mode = str(npz_time_mode).strip().lower()
        if mode not in {"absolute", "relative", "auto"}:
            raise ValueError(f"Invalid npz_time_mode={npz_time_mode}; expected absolute|relative|auto")
        if mode == "auto":
            absolute = _npz_time_is_absolute(ds, t_raw)
            mode_used = "npz:auto->absolute" if absolute else "npz:auto->relative"
        elif mode == "absolute":
            absolute = True
            mode_used = "npz:absolute"
        else:
            absolute = False
            mode_used = "npz:relative"

        if absolute:
            time_days = t_raw.astype(float)
            if apply_offset_to_npz and float(offset_days) != 0.0:
                time_days = time_days + float(offset_days)
                offset_applied = True
                mode_used = mode_used + "+offset"
        else:
            time_days = t_raw.astype(float) + float(offset_days)
            offset_applied = True
    elif str(path).endswith("staout"):
        data = np.loadtxt(str(path) + "_1")
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{path}_1 has unexpected shape")
        time_days = data[:, 0].astype(float) / 86400.0 + float(offset_days)
        elev = data[:, 1:].T.astype(float)
        mode_used = "staout:relative+offset"
        offset_applied = True
    else:
        raise ValueError(f"Unsupported run type: {path}")

    if elev.ndim == 1:
        elev = elev.reshape(1, -1)
    nt = len(time_days)
    if elev.shape[1] == nt:
        pass
    elif elev.shape[0] == nt:
        elev = elev.T
    else:
        raise ValueError(f"{path}: cannot align elev shape {elev.shape} with nt={nt}")

    return {
        "time": time_days,
        "elev": elev,
        "source_kind": source_kind,
        "time_mode_used": mode_used,
        "offset_days_used": float(offset_days) if offset_applied else 0.0,
    }


def _build_station_info(bpfile: Path) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    bp = read_schism_bpfile(str(bpfile))
    station_ids: List[str] = []
    station_vars: List[str] = []
    x = np.asarray(getattr(bp, "x"), dtype=float).ravel()
    y = np.asarray(getattr(bp, "y"), dtype=float).ravel()
    for entry in np.asarray(bp.station).astype("U").tolist():
        parts = str(entry).split()
        sid = parts[0] if parts else ""
        svar = parts[1].upper() if len(parts) > 1 else "WL"
        station_ids.append(sid)
        station_vars.append(svar)
    return station_ids, station_vars, x, y


def _load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)

    if args.runs:
        cfg["runs"] = list(args.runs)
    if args.tags:
        cfg["tags"] = list(args.tags)
    if args.bpfile:
        cfg["bpfile"] = args.bpfile
    if args.outdir:
        cfg["outdir"] = args.outdir
    if args.obs_path:
        cfg["obs_path"] = args.obs_path
    if args.start:
        cfg["start"] = args.start
    if args.end:
        cfg["end"] = args.end
    if args.resample_obs is not None:
        cfg["resample_obs"] = args.resample_obs
    if args.resample_model is not None:
        cfg["resample_model"] = args.resample_model
    if args.station_list:
        cfg["station_list"] = list(args.station_list)
    if args.demean is not None:
        cfg["demean"] = bool(args.demean)
    if args.save_plots is not None:
        cfg["save_plots"] = bool(args.save_plots)
    if args.filter_obs is not None:
        cfg["filter_obs"] = bool(args.filter_obs)
    if args.filter_model is not None:
        cfg["filter_model"] = bool(args.filter_model)
    if args.cutoff_period_hours is not None:
        cfg["cutoff_period_hours"] = float(args.cutoff_period_hours)
    if args.butterworth_order is not None:
        cfg["butterworth_order"] = int(args.butterworth_order)
    if args.line_width is not None:
        cfg["line_width"] = float(args.line_width)
    if args.plot_ymin is not None or args.plot_ymax is not None:
        ymin = args.plot_ymin if args.plot_ymin is not None else cfg.get("plot_ylim", [None, None])[0]
        ymax = args.plot_ymax if args.plot_ymax is not None else cfg.get("plot_ylim", [None, None])[1]
        cfg["plot_ylim"] = [ymin, ymax]
    if args.progress_every is not None:
        cfg["progress_every"] = int(args.progress_every)
    if args.grid:
        cfg["grid"] = args.grid
    if args.map_zoom is not None:
        cfg["map_zoom"] = float(args.map_zoom)
    if args.model_time_offset_days:
        cfg["model_time_offset_days"] = [float(v) for v in args.model_time_offset_days]
    elif args.model_start:
        cfg["model_time_offset_days"] = [float(datenum(args.model_start))]
    if args.npz_time_mode:
        cfg["npz_time_mode"] = str(args.npz_time_mode).strip().lower()
    if args.apply_offset_to_npz is not None:
        cfg["apply_offset_to_npz"] = bool(args.apply_offset_to_npz)

    runs = list(cfg.get("runs", []))
    tags = list(cfg.get("tags", []))
    if len(runs) == 0:
        raise ValueError("No runs configured.")
    if len(tags) == 1 and len(runs) > 1:
        tags = tags * len(runs)
    if len(tags) != len(runs):
        raise ValueError(f"tags length ({len(tags)}) must match runs length ({len(runs)}).")
    if "model_time_offset_days" not in cfg and cfg.get("model_start"):
        cfg["model_time_offset_days"] = [float(datenum(cfg["model_start"]))]
    elif cfg.get("model_start") and "model_time_offset_days" in cfg:
        # Respect explicit offsets when provided; otherwise keep model_start as metadata.
        pass

    offsets = _expand_scalar_or_list(cfg.get("model_time_offset_days", [0.0]), len(runs))
    mode = str(cfg.get("npz_time_mode", "absolute")).strip().lower()
    if mode not in {"absolute", "relative", "auto"}:
        raise ValueError(f"Invalid npz_time_mode={mode}; expected absolute|relative|auto")
    cfg["runs"] = runs
    cfg["tags"] = tags
    cfg["model_time_offset_days"] = [float(x) for x in offsets]
    cfg["npz_time_mode"] = mode
    cfg["apply_offset_to_npz"] = bool(cfg.get("apply_offset_to_npz", False))
    return cfg


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WL-only SCHISM vs JODC comparison.")
    p.add_argument("--config", help="JSON config file path.")
    p.add_argument("--runs", nargs="+", help="Model runs (.npz or staout prefix).")
    p.add_argument("--tags", nargs="+", help="Model labels matching --runs.")
    p.add_argument("--bpfile", help="SCHISM station bp file.")
    p.add_argument("--outdir", help="Output directory.")
    p.add_argument("--obs-path", help="Path to jodc_tide_all.npz.")
    p.add_argument("--start", help="Start time (string or datenum).")
    p.add_argument("--end", help="End time (string or datenum).")
    p.add_argument("--model-start", help="Model start datetime used as offset for model time.")
    p.add_argument(
        "--model-time-offset-days",
        nargs="+",
        type=float,
        help="Offset days to add to model time (one value or per-run values).",
    )
    p.add_argument(
        "--npz-time-mode",
        choices=["absolute", "relative", "auto"],
        help="How to interpret NPZ model time arrays.",
    )
    p.add_argument(
        "--apply-offset-to-npz",
        dest="apply_offset_to_npz",
        action="store_true",
        help="Also apply model_time_offset_days to NPZ even in absolute mode.",
    )
    p.add_argument(
        "--no-apply-offset-to-npz",
        dest="apply_offset_to_npz",
        action="store_false",
        help="Do not apply model_time_offset_days to NPZ in absolute mode.",
    )
    p.set_defaults(apply_offset_to_npz=None)
    p.add_argument("--resample-obs", help="Obs resample frequency (e.g., h, d).")
    p.add_argument("--resample-model", help="Model resample frequency (e.g., h, d).")
    p.add_argument("--station-list", nargs="+", help="Optional station IDs filter.")
    p.add_argument(
        "--demean",
        dest="demean",
        action="store_true",
        help="Subtract mean from obs/model before comparison.",
    )
    p.add_argument("--no-demean", dest="demean", action="store_false", help="Disable de-mean.")
    p.set_defaults(demean=None)
    p.add_argument("--save-plots", dest="save_plots", action="store_true", help="Write station plots.")
    p.add_argument("--no-save-plots", dest="save_plots", action="store_false", help="Skip plotting.")
    p.set_defaults(save_plots=None)
    p.add_argument("--filter-obs", dest="filter_obs", action="store_true", help="Low-pass filter obs.")
    p.add_argument("--no-filter-obs", dest="filter_obs", action="store_false", help="Do not filter obs.")
    p.set_defaults(filter_obs=None)
    p.add_argument("--filter-model", dest="filter_model", action="store_true", help="Low-pass filter model.")
    p.add_argument("--no-filter-model", dest="filter_model", action="store_false", help="Do not filter model.")
    p.set_defaults(filter_model=None)
    p.add_argument("--cutoff-period-hours", type=float, help="Low-pass cutoff period in hours.")
    p.add_argument("--butterworth-order", type=int, help="Butterworth filter order.")
    p.add_argument("--line-width", type=float, help="Plot line width.")
    p.add_argument("--plot-ymin", type=float, help="Y-axis minimum.")
    p.add_argument("--plot-ymax", type=float, help="Y-axis maximum.")
    p.add_argument("--progress-every", type=int, help="Progress print frequency by station count.")
    p.add_argument("--grid", help="SCHISM grid file for map panel (gr3).")
    p.add_argument("--map-zoom", type=float, help="Half-width in degrees around station for map panel.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args)

    outdir = _resolve_path(str(cfg["outdir"]))
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    if MPI:
        COMM.Barrier()

    cfg_snapshot = outdir / "wl_config_used.json"
    if RANK == 0:
        with open(cfg_snapshot, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    if MPI:
        COMM.Barrier()

    start_dnum = _parse_time_value(cfg["start"])
    end_dnum = _parse_time_value(cfg["end"])
    if end_dnum < start_dnum:
        raise ValueError("end must be >= start")

    obs_path = _resolve_path(str(cfg["obs_path"]))
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_path}")

    station_ids, station_vars, station_lon, station_lat = _build_station_info(_resolve_path(str(cfg["bpfile"])))
    wl_indices = [i for i, v in enumerate(station_vars) if str(v).upper() == "WL"]
    if not wl_indices:
        wl_indices = list(range(len(station_ids)))

    if cfg.get("station_list"):
        requested = {str(s).strip() for s in cfg["station_list"]}
        wl_indices = [
            i for i in wl_indices if station_ids[i] in requested or str(i + 1) in requested
        ]

    local_indices = [idx for pos, idx in enumerate(wl_indices) if (pos % SIZE) == RANK]
    _report_station_assignment("WL stations", len(wl_indices), local_indices)

    t0 = time.time()
    obs_map = _load_observations(obs_path)
    obs_lookup = _build_station_lookup(obs_map)
    log(f"Loaded observations for {len(obs_map)} stations", rank0_only=True)
    gd = None
    if cfg.get("grid"):
        grid_path = _resolve_path(str(cfg["grid"]))
        try:
            gd = read(str(grid_path))
        except Exception as exc:
            log(f"[WARN] Failed to read grid {grid_path}: {exc}", rank0_only=True)

    model_runs: List[Dict[str, Any]] = []
    for run_path, tag, offset in zip(cfg["runs"], cfg["tags"], cfg["model_time_offset_days"]):
        rp = _resolve_path(str(run_path))
        if not rp.exists() and not str(rp).endswith("staout"):
            raise FileNotFoundError(f"Model run not found: {rp}")
        model = _load_model_run(
            rp,
            float(offset),
            npz_time_mode=str(cfg.get("npz_time_mode", "absolute")),
            apply_offset_to_npz=bool(cfg.get("apply_offset_to_npz", False)),
        )
        model_runs.append({"path": str(rp), "tag": tag, **model})
        tmin = float(np.nanmin(model["time"])) if len(model["time"]) > 0 else np.nan
        tmax = float(np.nanmax(model["time"])) if len(model["time"]) > 0 else np.nan
        tmin_txt = num2date(tmin).strftime("%Y-%m-%d %H:%M:%S") if np.isfinite(tmin) else "nan"
        tmax_txt = num2date(tmax).strftime("%Y-%m-%d %H:%M:%S") if np.isfinite(tmax) else "nan"
        log(
            f"Loaded model {tag}: nsta={model['elev'].shape[0]}, nt={model['elev'].shape[1]}, "
            f"time={model.get('time_mode_used','')}, "
            f"range=[{tmin_txt} .. {tmax_txt}]",
            rank0_only=True,
        )

    rows_local: List[Dict[str, Any]] = []
    progress_every = max(1, int(cfg.get("progress_every", 20)))

    for p, sidx in enumerate(local_indices, start=1):
        sid = station_ids[sidx]
        obs_key = obs_lookup.get(_normalize_station_id(sid))
        if obs_key is None:
            obs_key = obs_lookup.get(_normalize_station_id(sid).lstrip("0"))
        if obs_key is None:
            continue
        oti_raw, oyi_raw = obs_map[obs_key]
        mask = (oti_raw >= start_dnum) & (oti_raw <= end_dnum)
        oti = oti_raw[mask]
        oyi = oyi_raw[mask]
        if len(oti) == 0:
            continue

        if cfg.get("filter_obs", False):
            oyi = _apply_lowpass(
                oyi,
                oti,
                float(cfg["cutoff_period_hours"]),
                int(cfg["butterworth_order"]),
            )
        obs_series = _resample(oti, oyi, cfg.get("resample_obs"))
        if len(obs_series) == 0:
            continue
        if cfg.get("demean", True):
            obs_mean = float(obs_series.mean())
            if np.isfinite(obs_mean):
                obs_series = obs_series - obs_mean

        fig = None
        ax_map = None
        ax_ts = None
        if cfg.get("save_plots", True):
            fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.0, 2.2]})
            ax_ts.plot(
                obs_series.index,
                obs_series.values,
                linestyle="None",
                marker="o",
                markersize=2.5,
                color="red",
                alpha=0.8,
                label="Obs",
            )

            # left panel: station location map
            if gd is not None:
                try:
                    gd.plot_bnd(ax=ax_map)
                except Exception as exc:
                    log(f"[WARN] grid boundary plotting failed: {exc}", rank0_only=True)
            finite_all = np.isfinite(station_lon) & np.isfinite(station_lat)
            if np.any(finite_all):
                ax_map.plot(station_lon[finite_all], station_lat[finite_all], ".", color="0.65", ms=3.0, label="Stations")
            lon0 = float(station_lon[sidx]) if sidx < len(station_lon) else np.nan
            lat0 = float(station_lat[sidx]) if sidx < len(station_lat) else np.nan
            if np.isfinite(lon0) and np.isfinite(lat0):
                ax_map.plot(lon0, lat0, "ro", ms=5, label="Obs site")
                dz = float(cfg.get("map_zoom", 0.1))
                if np.isfinite(dz) and dz > 0:
                    ax_map.set_xlim(lon0 - dz, lon0 + dz)
                    ax_map.set_ylim(lat0 - dz, lat0 + dz)
            ax_map.set_title(f"Station {sid}")
            ax_map.set_xlabel("Lon")
            ax_map.set_ylabel("Lat")
            ax_map.grid(alpha=0.3)
            ax_map.legend(loc="best", fontsize=8)

        text_lines = []
        for model in model_runs:
            if sidx >= model["elev"].shape[0]:
                continue
            mti = model["time"]
            myi = model["elev"][sidx, :].astype(float)
            mmask = (mti >= start_dnum) & (mti <= end_dnum)
            mti = mti[mmask]
            myi = myi[mmask]
            valid = np.isfinite(myi)
            mti = mti[valid]
            myi = myi[valid]
            if len(mti) == 0:
                continue
            if cfg.get("filter_model", False):
                myi = _apply_lowpass(
                    myi,
                    mti,
                    float(cfg["cutoff_period_hours"]),
                    int(cfg["butterworth_order"]),
                )

            model_series = _resample(mti, myi, cfg.get("resample_model"))
            if len(model_series) == 0:
                continue
            if cfg.get("demean", True):
                mod_mean = float(model_series.mean())
                if np.isfinite(mod_mean):
                    model_series = model_series - mod_mean

            if len(model_series.index) < 2:
                continue
            obs_clip = obs_series[
                (obs_series.index >= model_series.index.min())
                & (obs_series.index <= model_series.index.max())
            ]
            if len(obs_clip) == 0:
                continue
            interp_series = (
                model_series.reindex(model_series.index.union(obs_clip.index))
                .sort_index()
                .interpolate(method="time")
                .reindex(obs_clip.index)
            )
            valid = obs_clip.notna() & interp_series.notna()
            if valid.sum() < 2:
                continue
            obs_vals = obs_clip[valid].to_numpy(dtype=float)
            mod_vals = interp_series[valid].to_numpy(dtype=float)
            metrics = _compute_metrics(obs_vals, mod_vals)
            row = {
                "model": model["tag"],
                "station": sid,
                "station_index_1based": sidx + 1,
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
            rows_local.append(row)
            text_lines.append(
                f"{model['tag']}: R={metrics['corr']:.2f}, RMSE={metrics['rmse']:.3f}, "
                f"Bias={metrics['bias']:.3f}, WSS={metrics['wss']:.3f}"
            )

            if ax_ts is not None:
                ax_ts.plot(
                    model_series.index,
                    model_series.values,
                    linewidth=float(cfg["line_width"]),
                    label=model["tag"],
                )

        if ax_ts is not None and len(ax_ts.lines) > 1:
            ax_ts.set_title(f"WL station {sid}")
            ax_ts.set_xlabel("Time")
            ax_ts.set_ylabel("Water level (m)")
            if cfg.get("plot_ylim") is not None:
                y0, y1 = cfg["plot_ylim"]
                if y0 is not None and y1 is not None:
                    ax_ts.set_ylim(float(y0), float(y1))
            ax_ts.grid(alpha=0.3)
            ax_ts.legend(loc="best")
            if text_lines:
                ax_ts.text(
                    0.01,
                    0.02,
                    "\n".join(text_lines[:6]),
                    transform=ax_ts.transAxes,
                    va="bottom",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
            fig.tight_layout()
            fig.savefig(outdir / f"WL_station_{sid}.png", dpi=180)
        if fig is not None:
            plt.close(fig)

        if p == 1 or p % progress_every == 0 or p == len(local_indices):
            log(f"Processed {p}/{len(local_indices)} assigned stations")

    if MPI:
        all_rows = COMM.gather(rows_local, root=0)
    else:
        all_rows = [rows_local]

    if RANK == 0:
        flat_rows = [row for part in all_rows for row in part]
        df = pd.DataFrame(flat_rows)
        stats_csv = outdir / "WL_stats.csv"
        if len(df) > 0:
            df.sort_values(["model", "station"]).to_csv(stats_csv, index=False)
            by_model = []
            for model, sub in df.groupby("model"):
                w = sub["n"].astype(float).to_numpy()
                row = {
                    "model": model,
                    "stations": int(sub["station"].nunique()),
                    "samples": int(sub["n"].sum()),
                    "station_equal_bias": float(sub["bias"].mean()),
                    "station_equal_rmse": float(sub["rmse"].mean()),
                    "station_equal_corr": float(sub["corr"].mean()),
                    "station_equal_nrmse_std": float(sub["nrmse_std"].mean()),
                    "station_equal_wss": float(sub["wss"].mean()),
                    "sample_weight_bias": float(np.average(sub["bias"], weights=w)) if w.sum() > 0 else np.nan,
                    "sample_weight_rmse": float(np.average(sub["rmse"], weights=w)) if w.sum() > 0 else np.nan,
                    "sample_weight_corr": float(np.average(sub["corr"], weights=w)) if w.sum() > 0 else np.nan,
                    "sample_weight_nrmse_std": float(np.average(sub["nrmse_std"], weights=w))
                    if w.sum() > 0
                    else np.nan,
                    "sample_weight_wss": float(np.average(sub["wss"], weights=w)) if w.sum() > 0 else np.nan,
                }
                by_model.append(row)
            by_model_df = pd.DataFrame(by_model).sort_values("model")
            by_model_df.to_csv(outdir / "WL_stats_by_model.csv", index=False)
        else:
            df.to_csv(stats_csv, index=False)

        summary = {
            "status": "ok",
            "rows": int(len(df)),
            "models": sorted(df["model"].unique().tolist()) if len(df) > 0 else [],
            "stations_compared": int(df["station"].nunique()) if len(df) > 0 else 0,
            "runtime_sec": round(time.time() - t0, 2),
            "mpi_size": int(SIZE),
            "obs_path": str(obs_path),
        }
        with open(outdir / "WL_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"Wrote {stats_csv}", rank0_only=True)
        log(f"Done in {summary['runtime_sec']} s", rank0_only=True)


if __name__ == "__main__":
    main()
