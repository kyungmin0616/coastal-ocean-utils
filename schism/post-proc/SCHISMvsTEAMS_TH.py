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

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pylib import loadz, read_schism_bpfile, datenum, num2date, read


CONFIG: Dict[str, Any] = dict(
    teams_npz="/scratch2/08924/kmpark/post-proc/npz/sendai_d2_timeseries.npz",
    schism_npzs=[
        "/scratch2/08924/kmpark/post-proc/npz/RUN01d_SB_d2.npz",
        "/scratch2/08924/kmpark/post-proc/npz/RUN01e_SB_d2.npz",
        "/scratch2/08924/kmpark/post-proc/npz/RUN02a_SB_d2.npz",
    ],
    bpfile="station_sendai_d2.bp",
    outdir="./CompTEAMS_RUN01e02a_SB_d2",
    schism_labels=None,
    vars=["temp", "sal"],
    resample="H",
    start="2017-01-02",
    end="2017-12-31",
    depth=None,
    depth_tol=0.5,
    depth_policy="mean",  # mean | first
    model_time_units="datenum",  # datenum | seconds
    model_time_offset=datenum(2017, 1, 2),  # days
    station_list=None,
    debug_times=True,
    grid="../RUN01d/hgrid.gr3",
    map_zoom=0.1,
    save_plots=True,
    write_task_metrics=True,
    write_integrated_scatter=True,
    task_name="th",
    experiment_id=None,
    metrics_raw_name="TH_metrics_raw.csv",
    metrics_station_name="SCHISMvsTEAMS_stats.csv",
    metrics_model_name="TH_stats_by_model.csv",
    manifest_name="TH_manifest.json",
    scatter_alpha=0.65,
    scatter_size=9,
    scatter_cmap="viridis",
)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


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
    if not paths and cfg.get("schism_npz"):
        paths = [cfg.get("schism_npz")]
    labels = cfg.get("schism_labels")
    models: List[Dict[str, Any]] = []
    for i, p in enumerate(paths):
        ds = loadz(p)
        label = labels[i] if labels and i < len(labels) else Path(p).stem
        models.append(
            {
                "path": p,
                "label": label,
                "data": ds,
                "names": _get_schism_station_names(ds),
                "nsta": _infer_station_count(ds, ["temp", "sal", "salt"]),
            }
        )
    return models


def _teams_station_location(teams: Any, station_id: str) -> Tuple[Optional[float], Optional[float]]:
    sid = np.asarray(teams.station_id).astype(str)
    mask = sid == station_id
    if not mask.any():
        return None, None
    lat = np.asarray(teams.lat)[mask]
    lon = np.asarray(teams.lon)[mask]
    lat = lat[np.isfinite(lat)]
    lon = lon[np.isfinite(lon)]
    if lat.size == 0 or lon.size == 0:
        return None, None
    return float(np.nanmedian(lat)), float(np.nanmedian(lon))


def _teams_station_depth_mean(teams: Any, station_id: str) -> float:
    sid = np.asarray(teams.station_id).astype(str)
    mask = sid == station_id
    if not mask.any() or not hasattr(teams, "depth"):
        return float("nan")
    dep = np.asarray(teams.depth, dtype=float)[mask]
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
) -> Tuple[Optional[pd.DatetimeIndex], Optional[np.ndarray]]:
    sid = np.asarray(teams.station_id).astype(str)
    sid_norm = np.array([_normalize_station_id(x) for x in sid])
    target = _normalize_station_id(station_id)
    target_nz = target.lstrip("0")
    sid_nz = np.array([s.lstrip("0") for s in sid_norm])
    mask = (sid_norm == target) | (sid_nz == target_nz)
    if not mask.any():
        return None, None

    obs_var = _display_var_name(var)
    vals = np.asarray(getattr(teams, obs_var))
    times = np.asarray(teams.time)
    depths = np.asarray(getattr(teams, "depth", np.nan))

    vals = vals[mask]
    times = times[mask]
    depths = depths[mask] if depths is not None else None

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
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(mod)
    n = int(valid.sum())
    if n < 2:
        return {
            "n": n,
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


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _build_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = _deep_update(cfg, user_cfg)

    if args.teams:
        cfg["teams_npz"] = args.teams
    if args.schism:
        cfg["schism_npzs"] = list(args.schism)
    if args.schism_labels:
        cfg["schism_labels"] = list(args.schism_labels)
    if args.bp:
        cfg["bpfile"] = args.bp
    if args.outdir:
        cfg["outdir"] = args.outdir
    if args.vars:
        cfg["vars"] = list(args.vars)
    if args.resample:
        cfg["resample"] = args.resample
    if args.start:
        cfg["start"] = args.start
    if args.end:
        cfg["end"] = args.end
    if args.depth is not None:
        cfg["depth"] = float(args.depth)
    if args.depth_tol is not None:
        cfg["depth_tol"] = float(args.depth_tol)
    if args.depth_policy:
        cfg["depth_policy"] = args.depth_policy
    if args.model_time_units:
        cfg["model_time_units"] = args.model_time_units
    if args.model_time_offset is not None:
        cfg["model_time_offset"] = float(args.model_time_offset)
    if args.station_list:
        cfg["station_list"] = list(args.station_list)
    if args.debug_times:
        cfg["debug_times"] = True
    if args.grid:
        cfg["grid"] = args.grid
    if args.map_zoom is not None:
        cfg["map_zoom"] = float(args.map_zoom)
    if args.experiment_id:
        cfg["experiment_id"] = args.experiment_id
    if args.disable_plots:
        cfg["save_plots"] = False
    if args.disable_scatter:
        cfg["write_integrated_scatter"] = False
    if args.disable_metrics:
        cfg["write_task_metrics"] = False

    cfg["vars"] = [_normalize_var_name(v) for v in cfg.get("vars", [])]
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _build_runtime_config(args)

    outdir = Path(cfg["outdir"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "th_config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    teams = loadz(str(Path(cfg["teams_npz"]).expanduser()))
    schism_models = _load_schism_models(cfg)
    if not schism_models:
        print("[ERROR] No SCHISM NPZ provided.")
        return

    gd = None
    if cfg.get("grid"):
        try:
            gd = read(cfg["grid"])
        except Exception as exc:
            print(f"[WARN] Failed to read grid {cfg['grid']}: {exc}")

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

    station_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []

    summary = {
        "stations_requested": len(bp_names),
        "stations_with_obs": 0,
        "station_var_pairs": 0,
        "station_var_pairs_with_overlap": 0,
        "models": [m["label"] for m in schism_models],
    }

    for idx, name in enumerate(bp_names):
        sid_full = _extract_station_id(name)
        sid_short = _station_id_short(name)
        station_depth = _teams_station_depth_mean(teams, sid_short)

        station_has_obs = False
        for var in cfg["vars"]:
            summary["station_var_pairs"] += 1
            obs_times, obs_vals = _teams_station_series(
                teams,
                sid_short,
                var,
                depth=cfg["depth"],
                depth_tol=float(cfg["depth_tol"]),
                depth_policy=str(cfg["depth_policy"]),
            )
            if obs_times is None:
                if cfg.get("debug_times", False):
                    print(f"[WARN] No TEAMS data for station={sid_full}, var={var}")
                continue
            station_has_obs = True

            obs_series = _resample_series(pd.DatetimeIndex(obs_times), np.asarray(obs_vals, dtype=float), cfg["resample"])
            obs_series = _slice_time_window(obs_series, cfg.get("start"), cfg.get("end"))
            if len(obs_series) == 0:
                continue

            model_series_list: List[Tuple[str, pd.Series]] = []
            for model in schism_models:
                schism = model["data"]
                schism_names = model["names"]
                schism_nsta = model["nsta"]

                schism_match = _match_station(name, schism_names) if schism_names else None
                if schism_match is not None:
                    sta_idx = schism_names.index(schism_match)
                elif schism_nsta is not None and idx < schism_nsta:
                    sta_idx = idx
                else:
                    if cfg.get("debug_times", False):
                        print(f"[WARN] No SCHISM station match for {name} in {model['label']}")
                    continue

                try:
                    arr = _get_schism_var(schism, var)
                except KeyError as exc:
                    if cfg.get("debug_times", False):
                        print(f"[WARN] {exc}")
                    continue

                arr = _time_first(arr, len(getattr(schism, "time")))
                mod_vals = arr[:, sta_idx]
                mod_times = _as_datetime_index(
                    getattr(schism, "time"),
                    str(cfg["model_time_units"]),
                    float(cfg["model_time_offset"]),
                )
                mod_series = _resample_series(mod_times, mod_vals, cfg["resample"])
                mod_series = _slice_time_window(mod_series, cfg.get("start"), cfg.get("end"))
                model_series_list.append((str(model["label"]), mod_series))

            if not model_series_list:
                continue

            var_had_overlap = False
            for label, mod_series in model_series_list:
                common = obs_series.index.intersection(mod_series.index)
                if len(common) == 0:
                    if cfg.get("debug_times", False):
                        print(f"[WARN] No overlap for station={sid_full}, var={var}, model={label}")
                    continue

                var_had_overlap = True
                obs_c = obs_series.loc[common].values.astype(float)
                mod_c = mod_series.loc[common].values.astype(float)
                mm = _compute_metrics(obs_c, mod_c)

                station_rows.append(
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
                    raw_rows.append(
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
                summary["station_var_pairs_with_overlap"] += 1

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
            ax_ts.text(
                0.01,
                0.98,
                "See CSV for stats",
                transform=ax_ts.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
            ax_ts.legend()
            ax_ts.grid(True, alpha=0.3)

            if gd is not None:
                lat0, lon0 = _teams_station_location(teams, sid_short)
                if lat0 is None or lon0 is None:
                    ax_map.set_axis_off()
                else:
                    try:
                        gd.plot_bnd(ax=ax_map)
                    except Exception:
                        gd.plot_bnd()
                    ax_map.plot(lon0, lat0, marker="o", color="red", markersize=5)
                    dz = float(cfg.get("map_zoom", 0.1))
                    ax_map.set_xlim(lon0 - dz, lon0 + dz)
                    ax_map.set_ylim(lat0 - dz, lat0 + dz)
                    ax_map.set_xlabel("Lon")
                    ax_map.set_ylabel("Lat")
            else:
                ax_map.set_axis_off()

            fig.tight_layout()
            fname = f"{sid_full}_{name}_{_normalize_var_name(var)}.png".replace(" ", "_")
            plt.savefig(outdir / fname, dpi=200)
            plt.close(fig)

        if station_has_obs:
            summary["stations_with_obs"] += 1

    model_rows = _aggregate_model_metrics(raw_rows)

    stats_path = outdir / str(cfg.get("metrics_station_name", "SCHISMvsTEAMS_stats.csv"))
    raw_path = outdir / str(cfg.get("metrics_raw_name", "TH_metrics_raw.csv"))
    model_path = outdir / str(cfg.get("metrics_model_name", "TH_stats_by_model.csv"))

    if cfg.get("write_task_metrics", True):
        station_fields = [
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
        raw_fields = [
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
        _write_csv(stats_path, station_rows, station_fields)
        _write_csv(raw_path, raw_rows, raw_fields)
        _write_csv(model_path, model_rows, model_fields)
        print(f"Wrote stats to {stats_path}")
        print(f"Wrote raw pairs to {raw_path}")
        print(f"Wrote model summary to {model_path}")
    else:
        print("write_task_metrics=False: skipped CSV output.")

    scatter_files: List[str] = []
    if cfg.get("write_integrated_scatter", True):
        scatter_files = _write_integrated_scatter(raw_rows, cfg, outdir)
        for sf in scatter_files:
            print(f"Wrote scatter: {sf}")

    manifest = {
        "summary": {
            **summary,
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


if __name__ == "__main__":
    main()
