#!/usr/bin/env python3
"""
Convert JODC hourly tide observations to NPZ with UHSLC-like structure.

Output bundle fields:
  - time: float days (matplotlib date number)
  - elev: tidal height values
  - station: integer station_id (mapped from station code)
  - bp: zdata with station_id, station_name, lon, lat, country, resolution, nsta

Usage:
  python jodc_to_npz.py --station 0112
  python jodc_to_npz.py --all --start 2001-01-01 --end 2001-12-31
"""
from __future__ import annotations
import argparse
import os
from datetime import datetime, timedelta

import numpy as np
from pylib import zdata, savez, date2num

MISSING = 9999.0


def _parse_line(line: str):
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
        vals = (vals + [np.nan] * 24)[:24]
    return station, day, vals


def _load_station(folder: str, start: str | None, end: str | None, tz_offset_hours: float, scale: float):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
    times = []
    values = []
    station_code = None
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
                station_code = station_code or station
                for h, v in enumerate(vals):
                    ts = day + timedelta(hours=h) - timedelta(hours=tz_offset_hours)
                    times.append(ts)
                    values.append(v * scale if np.isfinite(v) else v)
    if len(times) == 0:
        return station_code, np.array([]), np.array([])
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
    return station_code, times, values


def _read_bp(bp_path: str):
    if not os.path.isfile(bp_path):
        return {}
    with open(bp_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 3:
        return {}
    try:
        nsta = int(lines[1].split()[0])
    except Exception:
        return {}
    meta = {}
    for ln in lines[2:2 + nsta]:
        parts = ln.split("!")
        left = parts[0].split()
        if len(left) < 4:
            continue
        try:
            sid = int(left[0])
            lon = float(left[1])
            lat = float(left[2])
        except Exception:
            continue
        name = parts[1].strip() if len(parts) > 1 else str(sid)
        meta[str(sid)] = {"lon": lon, "lat": lat, "name": name}
    return meta

def _build_bp(station_ids, station_names, bp_meta):
    bp = zdata()
    bp.station_id = np.array(station_ids, dtype=int)
    bp.station_name = np.array(station_names, dtype="U64")
    lons = []
    lats = []
    for sid in station_ids:
        meta = bp_meta.get(str(sid), {})
        lons.append(meta.get("lon", np.nan))
        lats.append(meta.get("lat", np.nan))
    bp.lon = np.array(lons, dtype=float)
    bp.lat = np.array(lats, dtype=float)
    bp.country = np.array([""] * len(station_ids), dtype="U1")
    bp.resolution = np.array(["hourly"] * len(station_ids), dtype="U16")
    bp.nsta = len(station_ids)
    return bp


def main():
    parser = argparse.ArgumentParser(description="Convert JODC tide data to NPZ.")
    parser.add_argument("--station", default=None, help="Single station code (e.g., 0112, 2003, MA11)")
    parser.add_argument("--all", action="store_true", help="Process all station folders")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--tz_offset", type=float, default=0.0, help="Hours to shift timestamps (local -> UTC)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (e.g., 0.01 for cm->m)")
    parser.add_argument("--out", default=None, help="Output NPZ filename")
    parser.add_argument("--outdir", default=".", help="Output directory")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    if args.all:
        stations = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    else:
        if not args.station:
            raise SystemExit("Provide --station or --all.")
        stations = [args.station]

    records_time = []
    records_value = []
    records_station = []
    bp_meta = _read_bp(os.path.join(base, "station_jodc.bp"))
    code_to_id = {}
    id_to_code = {}
    for sid, meta in bp_meta.items():
        name = meta.get("name", "")
        code = name.split()[0] if name else None
        if code:
            code_to_id[code] = int(sid)
            id_to_code[int(sid)] = code
    station_id_map = {}
    station_names = []

    for idx, code in enumerate(sorted(stations), start=1):
        folder = os.path.join(base, code)
        if not os.path.isdir(folder):
            continue
        station_code, times, values = _load_station(folder, args.start, args.end, args.tz_offset, args.scale)
        if times.size == 0:
            continue
        # Match station code to bp meta; fallback to sequential
        sid = code_to_id.get(code, idx)
        station_id_map[code] = sid
        station_names.append(code)
        tnum = date2num(list(times))
        records_time.extend(tnum)
        records_value.extend(values.tolist())
        records_station.extend([station_id_map[code]] * len(values))

    if len(records_time) == 0:
        raise SystemExit("No data found for the selection.")

    times = np.array(records_time, dtype=float)
    values = np.array(records_value, dtype=float)
    stations = np.array(records_station, dtype=int)
    order = np.argsort(times)

    bundle = zdata()
    bundle.time = times[order]
    bundle.elev = values[order]
    code_lookup = {sid: code for code, sid in station_id_map.items()}
    bundle.station = np.array([code_lookup.get(int(s), str(s)) for s in stations[order]], dtype="U16")
    # Keep station_name as station code (e.g., MA11, 0112, 2003)
    ordered_ids = sorted(set(station_id_map.values()))
    ordered_names = [id_to_code.get(sid, str(sid)) for sid in ordered_ids]
    bundle.bp = _build_bp(ordered_ids, ordered_names, bp_meta)

    os.makedirs(args.outdir, exist_ok=True)
    if args.out:
        out_path = os.path.join(args.outdir, args.out)
    else:
        tag = "all" if args.all else station_names[0]
        out_path = os.path.join(args.outdir, f"jodc_tide_{tag}.npz")
    savez(out_path, bundle)
    print(f"[OK] wrote NPZ: {out_path}")


if __name__ == "__main__":
    main()
