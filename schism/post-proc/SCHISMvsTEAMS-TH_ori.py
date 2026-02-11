#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare SCHISM time series against TEAMS model outputs (temperature/salinity).

Inputs
------
- TEAMS NPZ: TEAMS/npz/sendai_d2_timeseries.npz
  Required fields: time (datetime64), station_id, temp, sal, depth
- SCHISM NPZ: RUN01d.npz (extracted using the bp file)
  Expected fields: time (datenum days) and variables like temp/salt,
  plus station names (station/station_id).
- Station file: TEAMS/station_sendai_d2.in (SCHISM bp format)

Outputs
-------
- Per-station plots in outdir (PNG).
- Summary CSV with bias/rmse/corr for each station and variable.

Notes
-----
- TEAMS data are grouped by station_id (string like "0037"). If multiple
  depths exist, you can select a target depth with --depth and --depth-tol,
  otherwise the series is depth-averaged.
- SCHISM time units are assumed to be datenum days by default; use
  --model-time-units seconds if needed, with --model-time-offset (days).

Examples
--------
python SCHISMvsTEAMS-TH.py \
  --teams TEAMS/npz/sendai_d2_timeseries.npz \
  --schism RUN01d.npz \
  --bp TEAMS/station_sendai_d2.in \
  --outdir ./images/teams_compare \
  --vars temp sal --resample H

python SCHISMvsTEAMS-TH.py --depth 5 --depth-tol 0.5 --resample D
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pylib import loadz, read_schism_bpfile, datenum, num2date, read


CONFIG = dict(
    teams_npz="/scratch2/08924/kmpark/post-proc/npz/sendai_d2_timeseries.npz",
    schism_npzs=["/scratch2/08924/kmpark/post-proc/npz/RUN01d_SB_d2.npz","/scratch2/08924/kmpark/post-proc/npz/RUN01e_SB_d2.npz","/scratch2/08924/kmpark/post-proc/npz/RUN02a_SB_d2.npz"],
    bpfile="station_sendai_d2.bp",
    outdir="./CompTEAMS_RUN01e02a_SB_d2",
    schism_labels=None,  # optional list of labels matching schism_npzs
    vars=["temp", "sal"],
    resample="H",
    start="2017-01-02",
    end="2017-12-31",
    depth=None,
    depth_tol=0.5,
    depth_policy="mean",  # mean | first
    model_time_units="datenum",  # datenum | seconds
    model_time_offset=datenum(2017,1,2),  # days
    station_list=None,
    debug_times=True,
    grid="../RUN01d/hgrid.gr3",
    map_zoom=0.1,
)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare SCHISM outputs with TEAMS time series.")
    p.add_argument("--teams", help="Path to TEAMS NPZ.")
    p.add_argument("--schism", nargs="+", help="Path(s) to SCHISM NPZ.")
    p.add_argument("--schism-labels", nargs="+", help="Optional labels for SCHISM NPZs.")
    p.add_argument("--bp", help="Path to SCHISM bp file for station names.")
    p.add_argument("--outdir", help="Output directory for plots and CSV.")
    p.add_argument("--vars", nargs="+", choices=["temp", "sal"], help="Variables to compare.")
    p.add_argument("--resample", help="Resample frequency (e.g., H, D, M).")
    p.add_argument("--start", help="Start datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    p.add_argument("--end", help="End datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS).")
    p.add_argument("--depth", type=float, help="Target depth for TEAMS series (m).")
    p.add_argument("--depth-tol", type=float, help="Depth tolerance (m) for TEAMS selection.")
    p.add_argument("--depth-policy", choices=["mean", "first"], help="How to handle multiple depths.")
    p.add_argument("--model-time-units", choices=["datenum", "seconds"],
                   help="SCHISM time units.")
    p.add_argument("--model-time-offset", type=float,
                   help="Offset to add to SCHISM time (days).")
    p.add_argument("--station-list", nargs="+",
                   help="Station IDs or names to include (space-separated).")
    p.add_argument("--debug-times", action="store_true",
                   help="Print model/obs time ranges and overlap for each station/var.")
    p.add_argument("--grid", help="SCHISM grid file for boundary plot (gr3).")
    p.add_argument("--map-zoom", type=float,
                   help="Half-width of zoom box in degrees for map panel.")
    return p.parse_args(argv)


def _as_datetime_index(times, units, offset_days):
    if units == "seconds":
        base = pd.to_datetime("1970-01-01")
        stamps = base + pd.to_timedelta(np.asarray(times, dtype=float), unit="s")
    else:
        stamps = [num2date(t + offset_days) for t in np.asarray(times, dtype=float)]
        stamps = pd.to_datetime(stamps)
    return pd.DatetimeIndex(stamps)


def _resample_series(times, values, freq):
    s = pd.Series(values, index=times).sort_index()
    if freq:
        freq = str(freq).strip()
        if freq and freq[0].isalpha():
            freq = freq.lower()
        s = s.resample(freq).mean()
    return s.dropna()


def _station_names_from_bp(bpfile):
    names = []
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


def _normalize_station_id(sid):
    s = str(sid).strip()
    if s == "":
        return s
    return s


def _station_id_suffix(name):
    return ""


def _station_id_short(name):
    if not name:
        return ""
    return _normalize_station_id(name)


def _extract_station_id(name):
    return str(name).strip() if name is not None else ""


def _match_station(target_name, schism_names):
    if target_name in schism_names:
        return target_name
    target_id = _extract_station_id(target_name)
    matches = [n for n in schism_names if _extract_station_id(n) == target_id]
    if len(matches) == 1:
        return matches[0]
    return None


def _get_schism_station_names(ds):
    for key in ("station", "station_id", "sta", "stn", "name"):
        if hasattr(ds, key):
            arr = np.asarray(getattr(ds, key))
            return [str(x) for x in arr.tolist()]
    return []


def _get_schism_var(ds, var):
    if hasattr(ds, var):
        return np.asarray(getattr(ds, var))
    aliases = {"sal": "salt", "temp": "temp"}
    alt = aliases.get(var, None)
    if alt and hasattr(ds, alt):
        return np.asarray(getattr(ds, alt))
    raise KeyError(f"Variable {var} not found in SCHISM NPZ.")


def _load_schism_models(cfg):
    paths = cfg.get("schism_npzs") or []
    if not paths and cfg.get("schism_npz"):
        paths = [cfg.get("schism_npz")]
    labels = cfg.get("schism_labels")
    models = []
    for i, p in enumerate(paths):
        ds = loadz(p)
        label = None
        if labels and i < len(labels):
            label = labels[i]
        if label is None:
            label = Path(p).stem
        models.append({
            "path": p,
            "label": label,
            "data": ds,
            "names": _get_schism_station_names(ds),
            "nsta": _infer_station_count(ds, cfg["vars"] + ["temp", "salt", "sal"]),
        })
    return models

    if hasattr(ds, var):
        return np.asarray(getattr(ds, var))
    aliases = {"sal": "salt", "temp": "temp"}
    alt = aliases.get(var, None)
    if alt and hasattr(ds, alt):
        return np.asarray(getattr(ds, alt))
    raise KeyError(f"Variable {var} not found in SCHISM NPZ.")

def _infer_station_count(ds, fallback_vars):
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


def _time_first(arr, nt):
    if arr.ndim == 1:
        return arr.reshape(nt, 1)
    if arr.shape[0] == nt:
        return arr
    if arr.shape[-1] == nt:
        return np.swapaxes(arr, 0, -1)
    raise ValueError("Unable to align SCHISM array with time dimension.")


def _teams_station_location(teams, station_id):
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


def _teams_station_series(teams, station_id, var, depth=None, depth_tol=0.5, depth_policy="mean"):
    sid = np.asarray(teams.station_id).astype(str)
    sid_norm = np.array([_normalize_station_id(x) for x in sid])
    target = _normalize_station_id(station_id)
    target_nz = target.lstrip("0")
    sid_nz = np.array([s.lstrip("0") for s in sid_norm])
    mask = (sid_norm == target) | (sid_nz == target_nz)
    if not mask.any():
        return None, None
    vals = np.asarray(getattr(teams, var))
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

    return pd.to_datetime(times), vals


def _compute_stats(obs, mod):
    if len(obs) == 0 or len(mod) == 0:
        return np.nan, np.nan, np.nan
    bias = float(np.nanmean(mod - obs))
    rmse = float(np.sqrt(np.nanmean((mod - obs) ** 2)))
    if np.all(np.isfinite(obs)) and np.all(np.isfinite(mod)) and len(obs) > 1:
        corr = float(np.corrcoef(obs, mod)[0, 1])
    else:
        corr = np.nan
    return bias, rmse, corr


def main(argv=None):
    args = _parse_args(argv)
    cfg = dict(CONFIG)

    if args.teams:
        cfg["teams_npz"] = args.teams
    if args.schism:
        cfg["schism_npzs"] = args.schism
    if args.schism_labels:
        cfg["schism_labels"] = args.schism_labels
    if args.bp:
        cfg["bpfile"] = args.bp
    if args.outdir:
        cfg["outdir"] = args.outdir
    if args.vars:
        cfg["vars"] = args.vars
    if args.resample:
        cfg["resample"] = args.resample
    if args.start:
        cfg["start"] = args.start
    if args.end:
        cfg["end"] = args.end
    if args.depth is not None:
        cfg["depth"] = args.depth
    if args.depth_tol is not None:
        cfg["depth_tol"] = args.depth_tol
    if args.depth_policy:
        cfg["depth_policy"] = args.depth_policy
    if args.model_time_units:
        cfg["model_time_units"] = args.model_time_units
    if args.model_time_offset is not None:
        cfg["model_time_offset"] = args.model_time_offset
    if args.station_list:
        cfg["station_list"] = args.station_list
    if args.debug_times:
        cfg["debug_times"] = True
    if args.grid:
        cfg["grid"] = args.grid
    if args.map_zoom is not None:
        cfg["map_zoom"] = float(args.map_zoom)

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    teams = loadz(cfg["teams_npz"])
    schism_models = _load_schism_models(cfg)
    if not schism_models:
        print("[ERROR] No SCHISM NPZ provided.")
        return

    gd = None
    if cfg.get("grid"):
        try:
            gd = read(cfg["grid"])
        except Exception as e:
            print(f"[WARN] Failed to read grid {cfg['grid']}: {e}")

    bp_names = _station_names_from_bp(cfg["bpfile"])
    if cfg["debug_times"]:
        try:
            sid_vals = np.asarray(teams.station_id).astype(str)
            sid_vals = [s.strip() for s in sid_vals]
            uniq = sorted(set(sid_vals))
            print(f"[DEBUG] TEAMS station_id unique: {len(uniq)}")
            print(f"[DEBUG] TEAMS station_id sample: {uniq[:10]}")
        except Exception as e:
            print(f"[DEBUG] TEAMS station_id read failed: {e}")
        bp_ids = [_station_id_short(n) for n in bp_names]
        print(f"[DEBUG] BP station_id sample: {bp_ids[:10]}")
    if cfg["station_list"]:
        keep = set(str(x) for x in cfg["station_list"])
        filtered = []
        for i, n in enumerate(bp_names, start=1):
            sid_full = _extract_station_id(n)
            sid_short = _station_id_short(n)
            if n in keep or sid_full in keep or sid_short in keep or str(i) in keep:
                filtered.append(n)
        bp_names = filtered


    stats_rows = []

    for idx, name in enumerate(bp_names):
        sid_full = _extract_station_id(name)
        sid_short = _station_id_short(name)
        for var in cfg["vars"]:
            obs_times, obs_vals = _teams_station_series(
                teams, sid_short, var,
                depth=cfg["depth"], depth_tol=cfg["depth_tol"],
                depth_policy=cfg["depth_policy"],
            )
            if obs_times is None:
                print(f"[WARN] No TEAMS data for {sid_full}")
                continue

            obs_times = pd.DatetimeIndex(obs_times)
            if obs_times.tz is None:
                obs_times = obs_times.tz_localize("UTC")
            obs_series = _resample_series(obs_times, obs_vals, cfg["resample"])

            model_series_list = []
            for model in schism_models:
                schism = model["data"]
                schism_names = model["names"]
                schism_nsta = model["nsta"]

                schism_match = _match_station(name, schism_names) if schism_names else None
                if schism_match is not None:
                    sta_idx = schism_names.index(schism_match)
                else:
                    if schism_nsta is not None and idx < schism_nsta:
                        sta_idx = idx
                    else:
                        print(f"[WARN] No SCHISM station match for {name} in {model['label']}")
                        continue

                try:
                    arr = _get_schism_var(schism, var)
                except KeyError as e:
                    print(f"[WARN] {e}")
                    continue

                arr = _time_first(arr, len(getattr(schism, "time")))
                mod_vals = arr[:, sta_idx]
                mod_times = _as_datetime_index(
                    getattr(schism, "time"),
                    cfg["model_time_units"],
                    cfg["model_time_offset"],
                )

                if cfg["start"]:
                    t0 = pd.to_datetime(cfg["start"], utc=True)
                    mod_times = mod_times.tz_convert("UTC") if mod_times.tz is not None else mod_times.tz_localize("UTC")
                    mask = mod_times >= t0
                    mod_times = mod_times[mask]
                    mod_vals = mod_vals[mask]
                if cfg["end"]:
                    t1 = pd.to_datetime(cfg["end"], utc=True)
                    mod_times = mod_times.tz_convert("UTC") if mod_times.tz is not None else mod_times.tz_localize("UTC")
                    mask = mod_times <= t1
                    mod_times = mod_times[mask]
                    mod_vals = mod_vals[mask]

                mod_series = _resample_series(mod_times, mod_vals, cfg["resample"])
                model_series_list.append((model["label"], mod_series))

            if not model_series_list:
                continue

            fig, (ax_ts, ax_map) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [2.2, 1]})
            ax_ts.plot(obs_series.index, obs_series.values, label="TEAMS",
                       linestyle="None", marker="o", markersize=2.5, alpha=0.8)

            def _fmt(series):
                if len(series.index) == 0:
                    return "n=0"
                return (f"n={len(series.index)} "
                        f"{series.index.min()} -> {series.index.max()}")

            if cfg["debug_times"]:
                print(f"[DEBUG] {name} {var} TEAMS: {_fmt(obs_series)}")

            for label, mod_series in model_series_list:
                common = obs_series.index.intersection(mod_series.index)
                if cfg["debug_times"]:
                    print(f"[DEBUG] {name} {var} {label}: {_fmt(mod_series)}")
                    if len(common) > 0:
                        print(f"[DEBUG] {name} {var} {label} overlap: n={len(common)} "
                              f"{common.min()} -> {common.max()}")
                if len(common) == 0:
                    print(f"[WARN] No overlap for {name} ({var}) in {label}")
                    continue

                obs_c = obs_series.loc[common].values
                mod_c = mod_series.loc[common].values
                bias, rmse, corr = _compute_stats(obs_c, mod_c)

                stats_rows.append({
                    "model": label,
                    "station_name": name,
                    "station_id": sid_short,
                    "station_id_full": sid_full,
                    "var": var,
                    "n": len(common),
                    "bias": bias,
                    "rmse": rmse,
                    "corr": corr,
                })

                ax_ts.plot(mod_series.index, mod_series.values, label=label, lw=1.2)

            if len(ax_ts.lines) <= 1:
                print(f"[WARN] No overlap for {name} ({var})")
                plt.close(fig)
                continue

            ax_ts.set_title(f"{name} - {var}")
            if len(obs_series.index) > 0:
                ax_ts.set_xlim(obs_series.index.min(), obs_series.index.max())
            ax_ts.text(0.01, 0.98, "See CSV for stats", transform=ax_ts.transAxes,
                       ha="left", va="top", fontsize=9,
                       bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
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
                    ax_map.plot(lon0, lat0, marker='o', color='red', markersize=5)
                    dz = float(cfg.get('map_zoom', 0.1))
                    ax_map.set_xlim(lon0 - dz, lon0 + dz)
                    ax_map.set_ylim(lat0 - dz, lat0 + dz)
                    ax_map.set_xlabel('Lon')
                    ax_map.set_ylabel('Lat')
            else:
                ax_map.set_axis_off()

            fig.tight_layout()
            fname = f"{sid_full}_{name}_{var}.png".replace(" ", "_")
            plt.savefig(outdir / fname, dpi=200)
            plt.close()

    if stats_rows:
        df = pd.DataFrame(stats_rows)
        df.to_csv(outdir / "SCHISMvsTEAMS_stats.csv", index=False)
        print(f"Wrote stats to {outdir / 'SCHISMvsTEAMS_stats.csv'}")
    else:
        print("No stats generated; check station matching and time overlap.")


if __name__ == "__main__":
    main()
