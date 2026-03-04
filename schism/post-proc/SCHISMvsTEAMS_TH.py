#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare SCHISM time-series against TEAMS observations (temperature/salinity).

This refactored task script supports:
- direct JSON config via --config (model_evaluation.py-friendly)
- per-station time-series plots
- standardized station metrics CSV for campaign scorecards
- raw aligned pair CSV for auditing
- integrated cross-station scatter plots (model vs observation)
"""

from __future__ import annotations

# =============================================================================
# Configuration
# =============================================================================
CONFIG = dict(
    model={
        "runs": [
            "./npz/RUN01g_SB_D2.npz",
            "./npz/RUN03a_SB_D2.npz",
            "./npz/RUN04a_SB_D2.npz",
            "./npz/RUN04a_SB_D2.npz",

        ],
        "labels": ["RUN01g", "RUN03a", "RUN04a", "RUN05a"],
        "variables": ["temp", "sal"],
        "npz_time_mode": "absolute",  # absolute | relative | auto
        "apply_time_offset_to_npz": False,
        "time_units": "datenum",  # datenum | seconds
        "time_offset": "2017-01-02",  # datenum-compatible string or offset days
    },
    obs={
        "path": "./npz/sendai_d2_timeseries.npz",
    },
    stations={
        "bpfile": "station_sendai_d2.bp",  # station metadata file (order/names)
        "list": None,  # optional station subset (ids/names); None=all
        "depth": None,  # target TEAMS depth (m); None=use all depths
        "depth_tol": 0.5,  # depth match tolerance (m) when depth is set
        "depth_policy": "mean",  # mean|first handling for multiple records per time
    },
    time={
        "start": "2012-02-01",
        "end": "2014-12-31",
        "resample": {"obs": "H", "model": "H"},
    },
    map={
        "grid": "../run/RUN04a/hgrid.gr3",
        "zoom_deg": 0.1,
        "region": {
            "shapefile": "./shp/SOB.shp",  # set to None to skip shapefile filtering
            "use_shapefile": True,
            "subset_bbox": None,  # (lon_min, lon_max, lat_min, lat_max)
        },
    },
    output={
        "dir": "./CompObs/CompTEAMS_SBD2_01g03a04a05a",
        "save_plots": True,
        "write_task_metrics": True,
        "write_scatter_plots": True,
        "task_name": "th",
        "experiment_id": None,
        "metrics_raw_name": "TH_metrics_raw.csv",
        "metrics_station_name": "SCHISMvsTEAMS_stats.csv",
        "metrics_model_name": "TH_stats_by_model.csv",
        "manifest_name": "TH_manifest.json",
    },
    plot={
        "scatter_alpha": 0.65,
        "scatter_size": 9,
        "scatter_cmap": "viridis",
    },
    debug={
        "times": True,
    },
)

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pylib import (
    loadz,
    read_schism_bpfile,
    datenum,
    num2date,
    read,
    read_shapefile_data,
    deep_update_dict,
    init_mpi_runtime,
    rank_log,
    report_work_assignment,
    compute_skill_metrics,
    write_csv_rows,
    read_csv_rows,
)

TH_STATION_FIELDS = [
    "model",
    "station_name",
    "station_id",
    "station_id_full",
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

TH_RAW_FIELDS = [
    "task",
    "experiment_id",
    "model",
    "station_name",
    "station_id",
    "station_id_full",
    "var",
    "time",
    "obs",
    "model_value",
    "error",
    "station_depth",
]

TH_MODEL_FIELDS = [
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

# =============================================================================
# MPI setup
# =============================================================================
MPI, COMM, RANK, SIZE, USE_MPI = init_mpi_runtime(sys.argv)
SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# Core helpers
# =============================================================================
def rank_print(*args: Any, **kwargs: Any) -> None:
    rank0_only = bool(kwargs.pop("rank0_only", False))
    msg = " ".join(str(a) for a in args)
    rank_log(msg, rank=RANK, size=SIZE, rank0_only=rank0_only)


def _report_station_assignment(tag: str, total_count: int, local_indices: Sequence[int]) -> None:
    report_work_assignment(
        tag,
        total_count,
        local_indices,
        rank=RANK,
        size=SIZE,
        comm=COMM,
        mpi_enabled=bool(MPI),
        logger=rank_print,
    )


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    return deep_update_dict(base, override, merge_list_of_dicts=False)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SCHISM outputs with TEAMS time series.")
    p.add_argument("--config", help="Optional JSON config overrides.")
    p.add_argument("--teams", help="Path to TEAMS NPZ.")
    p.add_argument("--schism", nargs="+", help="Path(s) to SCHISM NPZ.")
    p.add_argument("--schism-labels", nargs="+", help="Optional labels for SCHISM NPZs.")
    p.add_argument("--bp", help="Path to SCHISM bp file for station names.")
    p.add_argument("--outdir", help="Output directory for plots and CSV.")
    p.add_argument("--vars", nargs="+", choices=["temp", "sal", "salt"], help="Variables to compare.")
    p.add_argument("--resample", help="Resample frequency (e.g., H, D, M).")
    p.add_argument("--start", help="Start datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    p.add_argument("--end", help="End datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    p.add_argument("--depth", type=float, help="Target depth for TEAMS series (m).")
    p.add_argument("--depth-tol", type=float, help="Depth tolerance (m) for TEAMS selection.")
    p.add_argument("--depth-policy", choices=["mean", "first"], help="How to handle multiple depths.")
    p.add_argument(
        "--npz-time-mode",
        choices=["absolute", "relative", "auto"],
        help="How to interpret SCHISM NPZ time arrays.",
    )
    p.add_argument(
        "--apply-offset-to-npz",
        dest="apply_offset_to_npz",
        action="store_true",
        help="Also apply model_time_offset to NPZ in absolute mode.",
    )
    p.add_argument(
        "--no-apply-offset-to-npz",
        dest="apply_offset_to_npz",
        action="store_false",
        help="Do not apply model_time_offset to NPZ in absolute mode.",
    )
    p.set_defaults(apply_offset_to_npz=None)
    p.add_argument("--model-time-units", choices=["datenum", "seconds"], help="SCHISM time units.")
    p.add_argument("--model-time-offset", type=float, help="Offset to add to SCHISM time (days).")
    p.add_argument("--station-list", nargs="+", help="Station IDs or names to include.")
    p.add_argument("--debug-times", action="store_true", help="Print model/obs overlap details.")
    p.add_argument("--grid", help="SCHISM grid file for boundary plot (gr3).")
    p.add_argument("--map-zoom", type=float, help="Half-width of zoom box in degrees for map panel.")
    p.add_argument("--experiment-id", help="Experiment ID for task output rows.")
    p.add_argument("--disable-plots", action="store_true", help="Skip per-station time-series plots.")
    p.add_argument("--disable-scatter", action="store_true", help="Skip integrated scatter plots.")
    p.add_argument("--disable-metrics", action="store_true", help="Skip writing standardized metrics CSVs.")
    return p.parse_args(argv)


def _normalize_var_name(var: str) -> str:
    v = str(var).strip().lower()
    if v in {"sal", "salt", "s"}:
        return "salt"
    if v in {"temp", "temperature", "t"}:
        return "temp"
    return v


def _display_var_name(var: str) -> str:
    return "sal" if _normalize_var_name(var) == "salt" else "temp"


def _as_datetime_index(times: Any, units: str, offset_days: float) -> pd.DatetimeIndex:
    if units == "seconds":
        base = pd.to_datetime("1970-01-01", utc=True)
        stamps = base + pd.to_timedelta(np.asarray(times, dtype=float), unit="s")
    else:
        stamps = [num2date(t + offset_days) for t in np.asarray(times, dtype=float)]
        stamps = pd.to_datetime(stamps, utc=True)
    if isinstance(stamps, pd.DatetimeIndex):
        return stamps
    return pd.DatetimeIndex(stamps)


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
    finite = np.asarray(time_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    return bool(np.nanmedian(finite) > 10000.0)


def _npz_time_to_datetime_index(
    ds: Any,
    time_values: Any,
    npz_time_mode: str,
    offset_days: float,
    apply_offset_to_npz: bool,
    fallback_units: str,
) -> pd.DatetimeIndex:
    t_raw = np.asarray(time_values, dtype=float)
    mode = str(npz_time_mode).strip().lower()
    if mode not in {"absolute", "relative", "auto"}:
        mode = "auto"

    if mode == "auto":
        is_absolute = _npz_time_is_absolute(ds, t_raw)
    elif mode == "absolute":
        is_absolute = True
    else:
        is_absolute = False

    if is_absolute:
        t_days = t_raw.astype(float)
        if apply_offset_to_npz and float(offset_days) != 0.0:
            t_days = t_days + float(offset_days)
    else:
        t_days = t_raw.astype(float) + float(offset_days)

    # If values still look non-datenum, fall back to legacy unit handling.
    finite = t_days[np.isfinite(t_days)]
    if finite.size > 0 and np.nanmedian(finite) < 10000.0:
        return _as_datetime_index(time_values, fallback_units, float(offset_days))

    stamps = [num2date(float(t)).strftime("%Y-%m-%d %H:%M:%S") for t in t_days]
    return pd.to_datetime(stamps, utc=True)


def _resample_series(times: pd.DatetimeIndex, values: np.ndarray, freq: Optional[str]) -> pd.Series:
    s = pd.Series(values, index=times).sort_index()
    if freq:
        freq = str(freq).strip()
        if freq and freq[0].isalpha():
            freq = freq.lower()
        s = s.resample(freq).mean()
    return s.dropna()


def _slice_time_window(series: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
    out = series
    if start:
        t0 = pd.to_datetime(start, utc=True)
        out = out[out.index >= t0]
    if end:
        t1 = pd.to_datetime(end, utc=True)
        out = out[out.index <= t1]
    return out


def _station_names_from_bp(bpfile: str) -> List[str]:
    names: List[str] = []
    try:
        with open(bpfile, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for ln in lines:
            if "#" not in ln:
                continue
            comment = ln.split("#", 1)[1].strip()
            if comment:
                names.append(comment)
        if names:
            return names
    except Exception:
        pass

    bp = read_schism_bpfile(bpfile)
    for entry in bp.station:
        text = entry.strip()
        if "#" in text:
            text = text.split("#", 1)[1].strip()
        names.append(text if text else entry.strip())
    return names


def _normalize_station_id(sid: Any) -> str:
    return str(sid).strip()


def _station_id_short(name: str) -> str:
    return _normalize_station_id(name)


def _extract_station_id(name: Any) -> str:
    return str(name).strip() if name is not None else ""


def _build_station_index_lookup(station_names: List[str]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    id_hits: Dict[str, List[int]] = {}
    for i, name in enumerate(station_names):
        key = str(name)
        lookup[key] = i
        sid = _extract_station_id(name)
        if sid:
            id_hits.setdefault(sid, []).append(i)
            id_hits.setdefault(sid.lstrip("0"), []).append(i)
    for sid, idxs in id_hits.items():
        if len(idxs) == 1:
            lookup[sid] = idxs[0]
    return lookup


def _build_teams_station_index(teams: Any) -> Dict[str, np.ndarray]:
    sid_raw = np.asarray(teams.station_id).astype(str)
    index: Dict[str, List[int]] = {}
    for i, sid in enumerate(sid_raw):
        sid_n = _normalize_station_id(sid)
        sid_nz = sid_n.lstrip("0")
        index.setdefault(sid_n, []).append(i)
        index.setdefault(sid_nz, []).append(i)
    out: Dict[str, np.ndarray] = {}
    for key, vals in index.items():
        out[key] = np.asarray(sorted(set(vals)), dtype=int)
    return out


def _teams_indices_for_station(teams: Any, station_id: str, station_index: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    target = _normalize_station_id(station_id)
    target_nz = target.lstrip("0")
    if station_index is not None:
        idx = station_index.get(target)
        if idx is None:
            idx = station_index.get(target_nz)
        if idx is not None:
            return np.asarray(idx, dtype=int)

    sid = np.asarray(teams.station_id).astype(str)
    sid_norm = np.array([_normalize_station_id(x) for x in sid])
    sid_nz = np.array([s.lstrip("0") for s in sid_norm])
    mask = (sid_norm == target) | (sid_nz == target_nz)
    return np.where(mask)[0].astype(int)


def _match_station(target_name: str, schism_names: List[str]) -> Optional[str]:
    if target_name in schism_names:
        return target_name
    target_id = _extract_station_id(target_name)
    matches = [n for n in schism_names if _extract_station_id(n) == target_id]
    if len(matches) == 1:
        return matches[0]
    return None


def _get_schism_station_names(ds: Any) -> List[str]:
    for key in ("station", "station_id", "sta", "stn", "name"):
        if hasattr(ds, key):
            arr = np.asarray(getattr(ds, key))
            return [str(x) for x in arr.tolist()]
    return []


def _get_schism_var(ds: Any, var: str) -> np.ndarray:
    var_norm = _normalize_var_name(var)
    candidates = [var, var_norm]
    if var_norm == "salt":
        candidates.extend(["sal", "salt"])
    elif var_norm == "temp":
        candidates.extend(["temp", "temperature"])
    for c in candidates:
        if hasattr(ds, c):
            return np.asarray(getattr(ds, c))
    raise KeyError(f"Variable {var} not found in SCHISM NPZ.")


def _infer_station_count(ds: Any, fallback_vars: Iterable[str]) -> Optional[int]:
    if not hasattr(ds, "time"):
        return None
    nt = len(getattr(ds, "time"))
    for var in fallback_vars:
        if not hasattr(ds, var):
            continue
        arr = np.asarray(getattr(ds, var))
        if arr.ndim != 2:
            continue
        if arr.shape[0] == nt:
            return arr.shape[1]
        if arr.shape[1] == nt:
            return arr.shape[0]
        return arr.shape[0]
    return None


def _time_first(arr: np.ndarray, nt: int) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(nt, 1)
    if arr.shape[0] == nt:
        return arr
    if arr.shape[-1] == nt:
        return np.swapaxes(arr, 0, -1)
    raise ValueError("Unable to align SCHISM array with time dimension.")


def _load_schism_models(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = list(cfg.get("schism_npzs") or [])
    labels = cfg.get("schism_labels")
    models: List[Dict[str, Any]] = []
    for i, p in enumerate(paths):
        ds = loadz(p)
        label = labels[i] if labels and i < len(labels) else Path(p).stem
        names = _get_schism_station_names(ds)
        time_index = _npz_time_to_datetime_index(
            ds,
            getattr(ds, "time"),
            str(cfg.get("npz_time_mode", "absolute")),
            float(cfg.get("model_time_offset", 0.0)),
            bool(cfg.get("apply_offset_to_npz", False)),
            str(cfg.get("model_time_units", "datenum")),
        )
        models.append(
            {
                "path": p,
                "label": label,
                "data": ds,
                "names": names,
                "station_lookup": _build_station_index_lookup(names),
                "time_index": time_index,
                "nsta": _infer_station_count(ds, ["temp", "sal", "salt"]),
            }
        )
    return models


def _teams_station_location(
    teams: Any,
    station_id: str,
    station_index: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    idx = _teams_indices_for_station(teams, station_id, station_index)
    if len(idx) == 0:
        return None, None
    lat = np.asarray(teams.lat)[idx]
    lon = np.asarray(teams.lon)[idx]
    lat = lat[np.isfinite(lat)]
    lon = lon[np.isfinite(lon)]
    if lat.size == 0 or lon.size == 0:
        return None, None
    return float(np.nanmedian(lat)), float(np.nanmedian(lon))


def _resolve_map_region(region_cfg: Dict[str, Any]) -> Dict[str, Any]:
    shapefile = region_cfg.get("shapefile")
    use_shapefile = bool(region_cfg.get("use_shapefile", bool(shapefile)))
    px = py = None
    if use_shapefile and shapefile:
        shp_path = Path(str(shapefile))
        if not shp_path.is_absolute():
            shp_path = (SCRIPT_DIR / shp_path).resolve()
        if shp_path.exists():
            try:
                bp = read_shapefile_data(str(shp_path))
                px, py = bp.xy.T
            except Exception as exc:
                rank_print(f"[WARN] Failed to read region shapefile {shp_path}: {exc}")
        else:
            rank_print(f"[WARN] Region shapefile not found: {shp_path}")
    return {"px": px, "py": py}


def _teams_station_depth_mean(
    teams: Any,
    station_id: str,
    station_index: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    idx = _teams_indices_for_station(teams, station_id, station_index)
    if len(idx) == 0 or not hasattr(teams, "depth"):
        return float("nan")
    dep = np.asarray(teams.depth, dtype=float)[idx]
    dep = dep[np.isfinite(dep)]
    if dep.size == 0:
        return float("nan")
    return float(np.nanmean(dep))


def _teams_station_series(
    teams: Any,
    station_id: str,
    var: str,
    depth: Optional[float] = None,
    depth_tol: float = 0.5,
    depth_policy: str = "mean",
    station_index: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Optional[pd.DatetimeIndex], Optional[np.ndarray]]:
    idx = _teams_indices_for_station(teams, station_id, station_index)
    if len(idx) == 0:
        return None, None

    obs_var = _display_var_name(var)
    vals = np.asarray(getattr(teams, obs_var))
    times = np.asarray(teams.time)
    depths = np.asarray(getattr(teams, "depth", np.nan))

    vals = vals[idx]
    times = times[idx]
    depths = depths[idx] if depths is not None else None

    if depth is not None and depths is not None:
        dmask = np.isfinite(depths) & (np.abs(depths - depth) <= depth_tol)
        if dmask.any():
            vals = vals[dmask]
            times = times[dmask]
        elif depth_policy == "first":
            pass

    if depth is None and depth_policy == "mean" and depths is not None:
        df = pd.DataFrame({"time": times, "val": vals})
        grouped = df.groupby("time")["val"].mean()
        times = grouped.index.values
        vals = grouped.values

    return pd.to_datetime(times, utc=True), np.asarray(vals, dtype=float)


def _compute_metrics(obs: np.ndarray, mod: np.ndarray) -> Dict[str, float]:
    return compute_skill_metrics(obs, mod, min_n=2)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    write_csv_rows(path, rows, fieldnames)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    return read_csv_rows(path)


def _aggregate_model_metrics(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in raw_rows:
        key = (str(r["model"]), str(r["var"]))
        grouped.setdefault(key, {"obs": [], "mod": []})
        grouped[key]["obs"].append(float(r["obs"]))
        grouped[key]["mod"].append(float(r["model_value"]))

    out: List[Dict[str, Any]] = []
    for (model, var), vals in sorted(grouped.items()):
        m = _compute_metrics(np.asarray(vals["obs"]), np.asarray(vals["mod"]))
        out.append(
            {
                "model": model,
                "task": "th",
                "var": var,
                "n": m["n"],
                "bias": m["bias"],
                "rmse": m["rmse"],
                "corr": m["corr"],
                "obs_std": m["obs_std"],
                "mod_std": m["mod_std"],
                "nrmse_std": m["nrmse_std"],
                "wss": m["wss"],
                "crmsd": m["crmsd"],
            }
        )
    return out


def _sanitize_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "model"


def _write_integrated_scatter(raw_rows: List[Dict[str, Any]], cfg: Dict[str, Any], outdir: Path) -> List[str]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    files: List[str] = []
    for model_name, rows in sorted(by_model.items()):
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
        vars_info = [("temp", "Temperature", "°C"), ("salt", "Salinity", "PSU")]
        mappable = None
        for ax, (var, title, unit) in zip(axs, vars_info):
            sub = [r for r in rows if str(r.get("var")) == var]
            if len(sub) == 0:
                ax.text(0.5, 0.5, f"No {var} pairs", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x = np.asarray([float(r.get("model_value", np.nan)) for r in sub], dtype=float)
            y = np.asarray([float(r.get("obs", np.nan)) for r in sub], dtype=float)
            c = np.asarray([float(r.get("station_depth", np.nan)) for r in sub], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                ax.text(0.5, 0.5, f"Not enough {var} pairs", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x = x[valid]
            y = y[valid]
            c = c[valid]
            use_depth_color = np.isfinite(c).any()
            if not use_depth_color:
                c = np.arange(len(x), dtype=float)

            mappable = ax.scatter(
                x,
                y,
                c=c,
                s=float(cfg.get("scatter_size", 9)),
                alpha=float(cfg.get("scatter_alpha", 0.65)),
                cmap=str(cfg.get("scatter_cmap", "viridis")),
                edgecolors="none",
            )

            lo = float(np.nanmin(np.r_[x, y]))
            hi = float(np.nanmax(np.r_[x, y]))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0
            pad = 0.03 * (hi - lo)
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], "k", lw=1.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(f"Model ({unit})")
            ax.set_ylabel(f"Observation ({unit})")
            ax.set_title(title)

            mm = _compute_metrics(y, x)
            ax.text(
                0.03,
                0.96,
                f"ME: {mm['bias']:.3f} {unit}\nRMSE: {mm['rmse']:.3f} {unit}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=axs, orientation="horizontal", fraction=0.08, pad=0.14)
            cbar.set_label("Station mean depth (m)")
        fig.suptitle(f"{model_name}: integrated TH comparison", fontsize=11)
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])

        fp = outdir / f"{_sanitize_name(model_name)}_TH_scatter.png"
        fig.savefig(fp, dpi=320, bbox_inches="tight")
        plt.close(fig)
        files.append(str(fp))
    return files


def _build_canonical_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = _deep_update(cfg, user_cfg)

    if args.teams:
        cfg.setdefault("obs", {})
        cfg["obs"]["path"] = args.teams
    if args.schism:
        cfg.setdefault("model", {})
        cfg["model"]["runs"] = list(args.schism)
    if args.schism_labels:
        cfg.setdefault("model", {})
        cfg["model"]["labels"] = list(args.schism_labels)
    if args.bp:
        cfg.setdefault("stations", {})
        cfg["stations"]["bpfile"] = args.bp
    if args.outdir:
        cfg.setdefault("output", {})
        cfg["output"]["dir"] = args.outdir
    if args.vars:
        cfg.setdefault("model", {})
        cfg["model"]["variables"] = list(args.vars)
    if args.resample:
        cfg.setdefault("time", {})
        cfg["time"].setdefault("resample", {})
        cfg["time"]["resample"]["obs"] = args.resample
        cfg["time"]["resample"]["model"] = args.resample
    if args.start:
        cfg.setdefault("time", {})
        cfg["time"]["start"] = args.start
    if args.end:
        cfg.setdefault("time", {})
        cfg["time"]["end"] = args.end
    if args.depth is not None:
        cfg.setdefault("stations", {})
        cfg["stations"]["depth"] = float(args.depth)
    if args.depth_tol is not None:
        cfg.setdefault("stations", {})
        cfg["stations"]["depth_tol"] = float(args.depth_tol)
    if args.depth_policy:
        cfg.setdefault("stations", {})
        cfg["stations"]["depth_policy"] = args.depth_policy
    if args.npz_time_mode:
        cfg.setdefault("model", {})
        cfg["model"]["npz_time_mode"] = str(args.npz_time_mode).strip().lower()
    if args.apply_offset_to_npz is not None:
        cfg.setdefault("model", {})
        cfg["model"]["apply_time_offset_to_npz"] = bool(args.apply_offset_to_npz)
    if args.model_time_units:
        cfg.setdefault("model", {})
        cfg["model"]["time_units"] = args.model_time_units
    if args.model_time_offset is not None:
        cfg.setdefault("model", {})
        cfg["model"]["time_offset"] = float(args.model_time_offset)
    if args.station_list:
        cfg.setdefault("stations", {})
        cfg["stations"]["list"] = list(args.station_list)
    if args.debug_times:
        cfg.setdefault("debug", {})
        cfg["debug"]["times"] = True
    if args.grid:
        cfg.setdefault("map", {})
        cfg["map"]["grid"] = args.grid
    if args.map_zoom is not None:
        cfg.setdefault("map", {})
        cfg["map"]["zoom_deg"] = float(args.map_zoom)
    if args.experiment_id:
        cfg.setdefault("output", {})
        cfg["output"]["experiment_id"] = args.experiment_id
    if args.disable_plots:
        cfg.setdefault("output", {})
        cfg["output"]["save_plots"] = False
    if args.disable_scatter:
        cfg.setdefault("output", {})
        cfg["output"]["write_scatter_plots"] = False
    if args.disable_metrics:
        cfg.setdefault("output", {})
        cfg["output"]["write_task_metrics"] = False
    return cfg


def _canonical_to_runtime_config(canonical_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(canonical_cfg.get("model", {}))
    obs_cfg = dict(canonical_cfg.get("obs", {}))
    station_cfg = dict(canonical_cfg.get("stations", {}))
    time_cfg = dict(canonical_cfg.get("time", {}))
    map_cfg = dict(canonical_cfg.get("map", {}))
    output_cfg = dict(canonical_cfg.get("output", {}))
    plot_cfg = dict(canonical_cfg.get("plot", {}))
    debug_cfg = dict(canonical_cfg.get("debug", {}))
    resample_cfg = dict(time_cfg.get("resample", {}))

    time_offset_raw = model_cfg.get("time_offset", 0.0)
    if time_offset_raw is None:
        model_time_offset = 0.0
    elif isinstance(time_offset_raw, str):
        model_time_offset = float(datenum(time_offset_raw))
    else:
        model_time_offset = float(time_offset_raw)
    npz_time_mode = str(model_cfg.get("npz_time_mode", "absolute")).strip().lower()
    if npz_time_mode not in {"absolute", "relative", "auto"}:
        npz_time_mode = "absolute"

    cfg: Dict[str, Any] = {
        "teams_npz": obs_cfg.get("path"),
        "schism_npzs": list(model_cfg.get("runs") or []),
        "schism_labels": model_cfg.get("labels"),
        "bpfile": station_cfg.get("bpfile"),
        "outdir": output_cfg.get("dir"),
        "vars": list(model_cfg.get("variables") or []),
        "resample": resample_cfg.get("obs", "H"),
        "start": time_cfg.get("start"),
        "end": time_cfg.get("end"),
        "depth": station_cfg.get("depth"),
        "depth_tol": float(station_cfg.get("depth_tol", 0.5)),
        "depth_policy": station_cfg.get("depth_policy", "mean"),
        "model_time_units": model_cfg.get("time_units", "datenum"),
        "model_time_offset": model_time_offset,
        "npz_time_mode": npz_time_mode,
        "apply_offset_to_npz": bool(model_cfg.get("apply_time_offset_to_npz", False)),
        "station_list": station_cfg.get("list"),
        "debug_times": bool(debug_cfg.get("times", False)),
        "grid": map_cfg.get("grid"),
        "map_zoom": float(map_cfg.get("zoom_deg", 0.1)),
        "map_region": dict(map_cfg.get("region", {})),
        "save_plots": bool(output_cfg.get("save_plots", True)),
        "write_task_metrics": bool(output_cfg.get("write_task_metrics", True)),
        "write_integrated_scatter": bool(output_cfg.get("write_scatter_plots", True)),
        "task_name": output_cfg.get("task_name", "th"),
        "experiment_id": output_cfg.get("experiment_id"),
        "metrics_raw_name": output_cfg.get("metrics_raw_name", "TH_metrics_raw.csv"),
        "metrics_station_name": output_cfg.get("metrics_station_name", "SCHISMvsTEAMS_stats.csv"),
        "metrics_model_name": output_cfg.get("metrics_model_name", "TH_stats_by_model.csv"),
        "manifest_name": output_cfg.get("manifest_name", "TH_manifest.json"),
        "scatter_alpha": float(plot_cfg.get("scatter_alpha", 0.65)),
        "scatter_size": float(plot_cfg.get("scatter_size", 9)),
        "scatter_cmap": str(plot_cfg.get("scatter_cmap", "viridis")),
    }
    cfg["vars"] = [_normalize_var_name(v) for v in cfg.get("vars", [])]
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    canonical_cfg = _build_canonical_config(args)
    cfg = _canonical_to_runtime_config(canonical_cfg)

    outdir = Path(cfg["outdir"]).expanduser().resolve()
    if RANK == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    if MPI:
        COMM.Barrier()

    if RANK == 0:
        with open(outdir / "th_config_used.json", "w", encoding="utf-8") as f:
            json.dump(canonical_cfg, f, indent=2)
    if MPI:
        COMM.Barrier()

    teams = loadz(str(Path(cfg["teams_npz"]).expanduser()))
    teams_station_index = _build_teams_station_index(teams)
    schism_models = _load_schism_models(cfg)
    if not schism_models:
        rank_print("[ERROR] No SCHISM NPZ provided.")
        return

    gd = None
    if cfg.get("grid"):
        try:
            gd = read(cfg["grid"])
        except Exception as exc:
            rank_print(f"[WARN] Failed to read grid {cfg['grid']}: {exc}")
    region = _resolve_map_region(cfg.get("map_region", {}))

    bp_names = _station_names_from_bp(cfg["bpfile"])
    if cfg["station_list"]:
        keep = {str(x) for x in cfg["station_list"]}
        filtered = []
        for i, n in enumerate(bp_names, start=1):
            sid_full = _extract_station_id(n)
            sid_short = _station_id_short(n)
            if n in keep or sid_full in keep or sid_short in keep or str(i) in keep:
                filtered.append(n)
        bp_names = filtered

    local_indices = [i for i in range(len(bp_names)) if (i % SIZE) == RANK]
    _report_station_assignment("station loop", len(bp_names), local_indices)

    station_rows_local: List[Dict[str, Any]] = []
    raw_rows_local: List[Dict[str, Any]] = []

    summary_local = {
        "stations_with_obs": 0,
        "station_var_pairs": 0,
        "station_var_pairs_with_overlap": 0,
    }

    for idx in local_indices:
        name = bp_names[idx]
        sid_full = _extract_station_id(name)
        sid_short = _station_id_short(name)
        station_depth = _teams_station_depth_mean(teams, sid_short, station_index=teams_station_index)
        lat0, lon0 = _teams_station_location(teams, sid_short, station_index=teams_station_index)

        station_has_obs = False
        for var in cfg["vars"]:
            summary_local["station_var_pairs"] += 1
            obs_times, obs_vals = _teams_station_series(
                teams,
                sid_short,
                var,
                depth=cfg["depth"],
                depth_tol=float(cfg["depth_tol"]),
                depth_policy=str(cfg["depth_policy"]),
                station_index=teams_station_index,
            )
            if obs_times is None:
                if cfg.get("debug_times", False):
                    rank_print(f"[WARN] No TEAMS data for station={sid_full}, var={var}")
                continue
            station_has_obs = True

            obs_series = _resample_series(pd.DatetimeIndex(obs_times), np.asarray(obs_vals, dtype=float), cfg["resample"])
            obs_series = _slice_time_window(obs_series, cfg.get("start"), cfg.get("end"))
            if len(obs_series) == 0:
                continue

            model_series_list: List[Tuple[str, pd.Series]] = []
            for model in schism_models:
                schism = model["data"]
                schism_lookup = model["station_lookup"]
                schism_nsta = model["nsta"]

                sta_idx = schism_lookup.get(name)
                if sta_idx is None:
                    sta_idx = schism_lookup.get(sid_full)
                if sta_idx is None and schism_nsta is not None and idx < schism_nsta:
                    sta_idx = idx
                if sta_idx is None:
                    if cfg.get("debug_times", False):
                        rank_print(f"[WARN] No SCHISM station match for {name} in {model['label']}")
                    continue

                try:
                    arr = _get_schism_var(schism, var)
                except KeyError as exc:
                    if cfg.get("debug_times", False):
                        rank_print(f"[WARN] {exc}")
                    continue

                arr = _time_first(arr, len(getattr(schism, "time")))
                mod_vals = arr[:, sta_idx]
                mod_series = _resample_series(model["time_index"], mod_vals, cfg["resample"])
                mod_series = _slice_time_window(mod_series, cfg.get("start"), cfg.get("end"))
                model_series_list.append((str(model["label"]), mod_series))

            if not model_series_list:
                continue

            var_had_overlap = False
            metrics_by_label: Dict[str, Dict[str, float]] = {}
            for label, mod_series in model_series_list:
                common = obs_series.index.intersection(mod_series.index)
                if len(common) == 0:
                    if cfg.get("debug_times", False):
                        rank_print(f"[WARN] No overlap for station={sid_full}, var={var}, model={label}")
                    continue

                var_had_overlap = True
                obs_c = obs_series.loc[common].values.astype(float)
                mod_c = mod_series.loc[common].values.astype(float)
                mm = _compute_metrics(obs_c, mod_c)
                metrics_by_label[label] = mm

                station_rows_local.append(
                    {
                        "model": label,
                        "station_name": str(name),
                        "station_id": sid_short,
                        "station_id_full": sid_full,
                        "var": _normalize_var_name(var),
                        "n": mm["n"],
                        "bias": mm["bias"],
                        "rmse": mm["rmse"],
                        "corr": mm["corr"],
                        "obs_std": mm["obs_std"],
                        "mod_std": mm["mod_std"],
                        "nrmse_std": mm["nrmse_std"],
                        "wss": mm["wss"],
                        "crmsd": mm["crmsd"],
                    }
                )

                task_name = str(cfg.get("task_name", "th"))
                exp_id = cfg.get("experiment_id") or label
                for t, o, m in zip(common, obs_c, mod_c):
                    raw_rows_local.append(
                        {
                            "task": task_name,
                            "experiment_id": exp_id,
                            "model": label,
                            "station_name": str(name),
                            "station_id": sid_short,
                            "station_id_full": sid_full,
                            "var": _normalize_var_name(var),
                            "time": str(t),
                            "obs": float(o),
                            "model_value": float(m),
                            "error": float(m - o),
                            "station_depth": float(station_depth),
                        }
                    )

            if var_had_overlap:
                summary_local["station_var_pairs_with_overlap"] += 1

            if not cfg.get("save_plots", True):
                continue

            fig, (ax_ts, ax_map) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.2, 1]})
            ax_ts.plot(obs_series.index, obs_series.values, label="TEAMS", linestyle="None", marker="o", markersize=2.5, alpha=0.8)

            plotted = 0
            for label, mod_series in model_series_list:
                common = obs_series.index.intersection(mod_series.index)
                if len(common) == 0:
                    continue
                ax_ts.plot(mod_series.index, mod_series.values, label=label, lw=1.2)
                plotted += 1

            if plotted == 0:
                plt.close(fig)
                continue

            ax_ts.set_title(f"{name} - {_normalize_var_name(var)}")
            if len(obs_series.index) > 0:
                ax_ts.set_xlim(obs_series.index.min(), obs_series.index.max())
            text_lines = []
            for label, _ in model_series_list:
                mm = metrics_by_label.get(label)
                if mm is None:
                    continue
                text_lines.append(
                    f"{label}: R={mm['corr']:.2f}, RMSE={mm['rmse']:.3f}, "
                    f"Bias={mm['bias']:.3f}, WSS={mm['wss']:.3f}"
                )
            if text_lines:
                ax_ts.text(
                    0.01,
                    0.02,
                    "\n".join(text_lines[:6]),
                    transform=ax_ts.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8.5,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )
            ax_ts.legend()
            ax_ts.grid(True, alpha=0.3)

            if lat0 is None or lon0 is None:
                ax_map.set_axis_off()
            else:
                has_region_bg = region.get("px") is not None and region.get("py") is not None
                if has_region_bg:
                    # If shapefile background exists, use it and skip grid boundary.
                    ax_map.plot(region["px"], region["py"], "k-", lw=0.8, alpha=0.8)
                elif gd is not None:
                    try:
                        gd.plot_bnd(ax=ax_map)
                    except Exception:
                        gd.plot_bnd()
                else:
                    ax_map.set_axis_off()
                    continue
                ax_map.plot(lon0, lat0, marker="o", color="red", markersize=5)
                dz = float(cfg.get("map_zoom", 0.1))
                ax_map.set_xlim(lon0 - dz, lon0 + dz)
                ax_map.set_ylim(lat0 - dz, lat0 + dz)
                ax_map.set_xlabel("Lon")
                ax_map.set_ylabel("Lat")

            fig.tight_layout()
            fname = f"{sid_full}_{name}_{_normalize_var_name(var)}.png".replace(" ", "_")
            plt.savefig(outdir / fname, dpi=200)
            plt.close(fig)

        if station_has_obs:
            summary_local["stations_with_obs"] += 1

    chunk_dir = outdir / ".th_rank_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    station_chunk_path = chunk_dir / f"station_rank_{RANK:04d}.csv"
    raw_chunk_path = chunk_dir / f"raw_rank_{RANK:04d}.csv"
    _write_csv(station_chunk_path, station_rows_local, TH_STATION_FIELDS)
    _write_csv(raw_chunk_path, raw_rows_local, TH_RAW_FIELDS)

    if MPI:
        summary_chunks = COMM.gather(summary_local, root=0)
        COMM.Barrier()
    else:
        summary_chunks = [summary_local]

    if RANK != 0:
        return

    if MPI:
        station_rows = []
        raw_rows = []
        for rr in range(SIZE):
            station_rows.extend(_read_csv_rows(chunk_dir / f"station_rank_{rr:04d}.csv"))
            raw_rows.extend(_read_csv_rows(chunk_dir / f"raw_rank_{rr:04d}.csv"))
    else:
        station_rows = station_rows_local
        raw_rows = raw_rows_local

    summary = {
        "stations_requested": len(bp_names),
        "stations_with_obs": int(sum(s["stations_with_obs"] for s in summary_chunks)),
        "station_var_pairs": int(sum(s["station_var_pairs"] for s in summary_chunks)),
        "station_var_pairs_with_overlap": int(sum(s["station_var_pairs_with_overlap"] for s in summary_chunks)),
        "models": [m["label"] for m in schism_models],
    }

    model_rows = _aggregate_model_metrics(raw_rows)

    stats_path = outdir / str(cfg.get("metrics_station_name", "SCHISMvsTEAMS_stats.csv"))
    raw_path = outdir / str(cfg.get("metrics_raw_name", "TH_metrics_raw.csv"))
    model_path = outdir / str(cfg.get("metrics_model_name", "TH_stats_by_model.csv"))

    if cfg.get("write_task_metrics", True):
        _write_csv(stats_path, station_rows, TH_STATION_FIELDS)
        _write_csv(raw_path, raw_rows, TH_RAW_FIELDS)
        _write_csv(model_path, model_rows, TH_MODEL_FIELDS)
        rank_print(f"Wrote stats to {stats_path}")
        rank_print(f"Wrote raw pairs to {raw_path}")
        rank_print(f"Wrote model summary to {model_path}")
    else:
        rank_print("write_task_metrics=False: skipped CSV output.")

    scatter_files: List[str] = []
    if cfg.get("write_integrated_scatter", True):
        scatter_files = _write_integrated_scatter(raw_rows, cfg, outdir)
        for sf in scatter_files:
            rank_print(f"Wrote scatter: {sf}")

    manifest = {
        "summary": {
            **summary,
            "mpi_size": int(SIZE),
            "station_rows": len(station_rows),
            "raw_rows": len(raw_rows),
            "model_rows": len(model_rows),
        },
        "files": {
            "station_stats": str(stats_path) if cfg.get("write_task_metrics", True) else None,
            "raw_pairs": str(raw_path) if cfg.get("write_task_metrics", True) else None,
            "model_stats": str(model_path) if cfg.get("write_task_metrics", True) else None,
            "scatter": scatter_files,
        },
    }
    manifest_path = outdir / str(cfg.get("manifest_name", "TH_manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for fp in chunk_dir.glob("*rank_*.csv"):
        try:
            fp.unlink()
        except Exception:
            pass
    try:
        chunk_dir.rmdir()
    except Exception:
        pass


if __name__ == "__main__":
    main()
