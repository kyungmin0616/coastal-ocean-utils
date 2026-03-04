#!/usr/bin/env python3
"""
Plot JODC tide observation time series from an NPZ bundle (e.g., jodc_tide_all.npz).

Selection options:
1) explicit station list (--stations / config["stations"])
2) station list from a SCHISM bp file (--bpfile / config["bpfile"])
3) fallback: all stations in the NPZ
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylib import datenum, loadz, num2date, read_schism_bpfile


USER_CONFIG: Dict[str, Any] = {
    "enable": True,
    "npz_path": "./jodc_tide_all.npz",
    "outdir": "./plots_jodc_npz",
    "start": "2013-03-01 00:00:00",  # e.g. "2012-03-01 00:00:00"
    "end": "2013-03-14 00:00:00",  # e.g. "2012-03-14 00:00:00"
    "stations": None,  # e.g. ["MA11", "0112", "2003"]
    "bpfile": "./station_jodc.bp",  # e.g. "/scratch2/.../station_jodc.bp"
    "bp_wl_only": True,  # use only stations marked WL in bp
    "plot_mode": "hourly",  # hourly | daily | both
    "dpi": 220,
}


def _resolve_path(path_like: str) -> Path:
    p = Path(os.path.expanduser(str(path_like)))
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _normalize_station_id(sid: Any) -> str:
    return str(sid).strip()


def _parse_time_value(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    return float(datenum(str(value)))


def _time_range_text(times: np.ndarray) -> Tuple[str, str, int]:
    arr = np.asarray(times, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "nan", "nan", 0
    tmin = float(np.nanmin(arr))
    tmax = float(np.nanmax(arr))
    return (
        num2date(tmin).strftime("%Y-%m-%d %H:%M:%S"),
        num2date(tmax).strftime("%Y-%m-%d %H:%M:%S"),
        int(arr.size),
    )


def _load_npz_observations(npz_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    ds = loadz(str(npz_path))
    if not hasattr(ds, "station") or not hasattr(ds, "time") or not hasattr(ds, "elev"):
        raise ValueError(f"{npz_path}: expected keys station, time, elev")

    stations = np.asarray(ds.station).astype("U")
    times_raw = np.asarray(ds.time)
    elev = np.asarray(ds.elev, dtype=float)

    if np.issubdtype(times_raw.dtype, np.datetime64):
        tstr = times_raw.astype("datetime64[s]").astype("U")
        times = np.array([float(datenum(s)) for s in tstr], dtype=float)
    else:
        times = np.asarray(times_raw, dtype=float)

    df = pd.DataFrame({"station": stations, "time": times, "elev": elev})
    df = df[np.isfinite(df["time"]) & np.isfinite(df["elev"])]
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sid, grp in df.groupby("station"):
        t = grp["time"].to_numpy(dtype=float)
        v = grp["elev"].to_numpy(dtype=float)
        order = np.argsort(t)
        out[_normalize_station_id(sid)] = (t[order], v[order])
    return out


def _station_lookup(obs_map: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for key in obs_map:
        kn = _normalize_station_id(key)
        lookup[kn] = key
        lookup[kn.lstrip("0")] = key
    return lookup


def _stations_from_bp(bpfile: Path, wl_only: bool = True) -> List[str]:
    bp = read_schism_bpfile(str(bpfile))
    out: List[str] = []
    for entry in np.asarray(bp.station).astype("U").tolist():
        parts = str(entry).split()
        if len(parts) == 0:
            continue
        sid = parts[0]
        svar = parts[1].upper() if len(parts) > 1 else "WL"
        if wl_only and svar != "WL":
            continue
        out.append(_normalize_station_id(sid))
    # unique, preserve order
    seen = set()
    uniq = []
    for sid in out:
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append(sid)
    return uniq


def _to_hourly_series(times: np.ndarray, values: np.ndarray) -> pd.Series:
    if len(times) == 0:
        return pd.Series(dtype=float)
    stamps = [num2date(float(t)).strftime("%Y-%m-%d %H:%M:%S") for t in times]
    s = pd.Series(values.astype(float), index=pd.to_datetime(stamps, utc=True)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.dropna()


def _plot_series(
    series: pd.Series,
    title: str,
    ylab: str,
    outpath: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(11.0, 4.2))
    ax.plot(series.index, series.values, lw=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(USER_CONFIG) if USER_CONFIG.get("enable", False) else {}
    if args.npz_path is not None:
        cfg["npz_path"] = args.npz_path
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.start is not None:
        cfg["start"] = args.start
    if args.end is not None:
        cfg["end"] = args.end
    if args.stations is not None:
        cfg["stations"] = list(args.stations)
    if args.bpfile is not None:
        cfg["bpfile"] = args.bpfile
    if args.bp_wl_only is not None:
        cfg["bp_wl_only"] = bool(args.bp_wl_only)
    if args.plot_mode is not None:
        cfg["plot_mode"] = args.plot_mode
    if args.dpi is not None:
        cfg["dpi"] = int(args.dpi)

    if cfg.get("npz_path") is None:
        raise ValueError("npz_path is required.")
    if cfg.get("outdir") is None:
        raise ValueError("outdir is required.")
    mode = str(cfg.get("plot_mode", "both")).strip().lower()
    if mode not in {"hourly", "daily", "both"}:
        raise ValueError("plot_mode must be one of: hourly, daily, both")
    cfg["plot_mode"] = mode
    cfg["bp_wl_only"] = bool(cfg.get("bp_wl_only", True))
    cfg["dpi"] = int(cfg.get("dpi", 220))
    return cfg


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot JODC tide observations from NPZ.")
    p.add_argument("--npz-path", help="Path to jodc_tide_all.npz")
    p.add_argument("--outdir", help="Output directory for figures")
    p.add_argument("--start", help="Start time (string or datenum)")
    p.add_argument("--end", help="End time (string or datenum)")
    p.add_argument("--stations", nargs="+", help="Station IDs/codes to plot")
    p.add_argument("--bpfile", help="station_jodc.bp path for station selection")
    p.add_argument("--bp-wl-only", dest="bp_wl_only", action="store_true", help="Use only WL stations from bpfile")
    p.add_argument("--no-bp-wl-only", dest="bp_wl_only", action="store_false", help="Use all stations in bpfile")
    p.set_defaults(bp_wl_only=None)
    p.add_argument("--plot-mode", choices=["hourly", "daily", "both"], help="Which plots to write")
    p.add_argument("--dpi", type=int, help="Figure dpi")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _merge_config(args)

    npz_path = _resolve_path(str(cfg["npz_path"]))
    outdir = _resolve_path(str(cfg["outdir"]))
    outdir.mkdir(parents=True, exist_ok=True)

    start_dnum = _parse_time_value(cfg.get("start"))
    end_dnum = _parse_time_value(cfg.get("end"))
    if np.isfinite(start_dnum) and np.isfinite(end_dnum) and end_dnum < start_dnum:
        raise ValueError("end must be >= start")

    obs_map = _load_npz_observations(npz_path)
    lookup = _station_lookup(obs_map)
    print(f"Loaded NPZ observations: nsta={len(obs_map)}", flush=True)

    # Coverage summary for NPZ
    global_times = []
    for t, _ in obs_map.values():
        global_times.append(np.asarray(t, dtype=float))
    if len(global_times) > 0:
        t_all = np.concatenate(global_times)
        tmin_txt, tmax_txt, n_all = _time_range_text(t_all)
        print(f"Observation global coverage: n={n_all}, range=[{tmin_txt} .. {tmax_txt}]", flush=True)

    # Station selection priority: stations > bpfile > all
    selected: List[str]
    if cfg.get("stations"):
        selected = [_normalize_station_id(s) for s in cfg["stations"]]
        source = "stations list"
    elif cfg.get("bpfile"):
        bpfile = _resolve_path(str(cfg["bpfile"]))
        selected = _stations_from_bp(bpfile, wl_only=bool(cfg.get("bp_wl_only", True)))
        source = f"bpfile ({bpfile})"
    else:
        selected = sorted(obs_map.keys())
        source = "all stations in npz"
    print(f"Station selection source: {source}; requested={len(selected)}", flush=True)

    summary_rows: List[Dict[str, Any]] = []
    written = 0
    skipped = 0

    for sid in selected:
        key = lookup.get(_normalize_station_id(sid))
        if key is None:
            key = lookup.get(_normalize_station_id(sid).lstrip("0"))
        if key is None:
            print(f"[SKIP] station={sid} reason=not_in_npz", flush=True)
            skipped += 1
            continue

        t_raw, v_raw = obs_map[key]
        t = np.asarray(t_raw, dtype=float)
        v = np.asarray(v_raw, dtype=float)
        mask = np.isfinite(t) & np.isfinite(v)
        if np.isfinite(start_dnum):
            mask &= t >= start_dnum
        if np.isfinite(end_dnum):
            mask &= t <= end_dnum
        t = t[mask]
        v = v[mask]

        if len(t) == 0:
            print(f"[SKIP] station={sid} matched={key} reason=no_data_in_time_window", flush=True)
            skipped += 1
            continue

        station_tag = key.replace("/", "_")
        hourly = _to_hourly_series(t, v)
        tmin_txt, tmax_txt, npts = _time_range_text(t)
        print(f"Loaded observation {sid} -> key={key}: n={npts}, range=[{tmin_txt} .. {tmax_txt}]", flush=True)

        if cfg["plot_mode"] in {"hourly", "both"} and len(hourly) > 0:
            fp = outdir / f"JODC_{station_tag}_hourly.png"
            _plot_series(
                hourly,
                title=f"JODC Tide Observation (hourly) - {key}",
                ylab="Water Level",
                outpath=fp,
                dpi=cfg["dpi"],
            )
            written += 1

        if cfg["plot_mode"] in {"daily", "both"} and len(hourly) > 0:
            daily = hourly.resample("D").mean().dropna()
            if len(daily) > 0:
                fp = outdir / f"JODC_{station_tag}_daily_mean.png"
                _plot_series(
                    daily,
                    title=f"JODC Tide Observation (daily mean) - {key}",
                    ylab="Water Level",
                    outpath=fp,
                    dpi=cfg["dpi"],
                )
                written += 1

        summary_rows.append(
            {
                "requested_station": sid,
                "matched_station": key,
                "n_points": npts,
                "tmin": tmin_txt,
                "tmax": tmax_txt,
            }
        )

    if len(summary_rows) > 0:
        pd.DataFrame(summary_rows).to_csv(outdir / "jodc_npz_station_coverage.csv", index=False)
    with open(outdir / "jodc_npz_plot_config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"Done. stations_requested={len(selected)}, stations_plotted={len(summary_rows)}, "
        f"figures_written={written}, stations_skipped={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
