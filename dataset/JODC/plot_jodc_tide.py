#!/usr/bin/env python3
"""
Plot JODC tide observations (hourly tidal height).

Usage:
  python plot_jodc_tide.py --station 0112
  python plot_jodc_tide.py --station MA11 --start 1962-01-01 --end 1962-12-31
  python plot_jodc_tide.py --station 2003 --year 2018 --outdir ./plots
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

MISSING = 9999.0

def _parse_line(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    station = parts[0]
    try:
        day = datetime.strptime(parts[1], "%Y/%m/%d")
    except Exception:
        return None
    vals = []
    for v in parts[2:]:
        try:
            fv = float(v)
        except Exception:
            fv = np.nan
        if fv == MISSING:
            fv = np.nan
        vals.append(fv)
    if len(vals) != 24:
        # pad or trim to 24
        vals = (vals + [np.nan] * 24)[:24]
    return station, day, vals

def _load_station(folder, year=None, start=None, end=None):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    if year is not None:
        files = [f for f in files if f"_{year}" in f]
    times = []
    values = []
    station_id = None
    for fname in files:
        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.lower().startswith("moved"):
                    continue
                parsed = _parse_line(line)
                if parsed is None:
                    continue
                station, day, vals = parsed
                station_id = station_id or station
                for h, v in enumerate(vals):
                    ts = day + timedelta(hours=h)
                    times.append(ts)
                    values.append(v)
    if len(times) == 0:
        return station_id, np.array([]), np.array([])
    times = np.array(times)
    values = np.array(values, dtype=float)
    if start:
        t0 = datetime.strptime(start, "%Y-%m-%d")
        mask = times >= t0
        times = times[mask]
        values = values[mask]
    if end:
        t1 = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
        mask = times < t1
        times = times[mask]
        values = values[mask]
    return station_id, times, values

def _daily_mean(times, values):
    if times.size == 0:
        return np.array([]), np.array([])
    days = np.array([datetime(t.year, t.month, t.day) for t in times])
    uniq = np.unique(days)
    dvals = []
    for d in uniq:
        mask = days == d
        v = values[mask]
        v = v[~np.isnan(v)]
        dvals.append(np.nan if v.size == 0 else np.mean(v))
    return uniq, np.array(dvals, dtype=float)

def _plot_series(times, values, title, outpath):
    plt.figure(figsize=(12, 4))
    plt.plot(times, values, lw=0.6)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Tide height")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot JODC hourly tidal height observations.")
    parser.add_argument("--station", required=True, help="Station folder name (e.g., 0112, 2003, MA11)")
    parser.add_argument("--year", type=int, default=None, help="Filter to a single year")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--outdir", default=".", help="Output directory for plots")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base, args.station)
    if not os.path.isdir(folder):
        raise SystemExit(f"Station folder not found: {folder}")

    station_id, times, values = _load_station(folder, year=args.year, start=args.start, end=args.end)
    if times.size == 0:
        raise SystemExit("No data found for the selection.")

    os.makedirs(args.outdir, exist_ok=True)
    tag = station_id or args.station
    suffix = f"_{args.year}" if args.year else ""

    _plot_series(times, values, f"{tag} hourly tide{suffix}", os.path.join(args.outdir, f"{tag}_hourly{suffix}.png"))
    dtime, dvals = _daily_mean(times, values)
    _plot_series(dtime, dvals, f"{tag} daily mean tide{suffix}", os.path.join(args.outdir, f"{tag}_daily{suffix}.png"))

if __name__ == "__main__":
    main()
