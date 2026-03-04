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
from pathlib import Path

import numpy as np
from pylib import zdata, savez, date2num

MISSING = 9999.0


# =====================
# Config
# =====================
USER_CONFIG = {
    "enable": True,
    # If None, defaults to this script directory.
    "base_dir": None,
    # Station selection priority:
    # stations (list) > station (single) > all (bool)
    "stations": None,  # e.g. ["0112", "2003", "MA11"]
    "station": None,  # e.g. "0112"
    "all": True,
    # If True, derive station code list from bpfile and convert only those stations.
    "use_bp_stations": False,
    # When using bp stations, keep only entries with var token "WL" in bp comment (e.g. "0112 WL").
    "bp_wl_only": True,
    # Time window (inclusive dates, YYYY-MM-DD)
    "start": None,
    "end": None,
    # Data transform
    "tz_offset": 9.0,  # local -> UTC hours
    "scale": 0.01,  # e.g. 0.01 for cm->m
    # Output
    "out": None,  # explicit output filename
    "outdir": ".",
    # Optional bp path for station metadata. If None: {base_dir}/station_jodc.bp
    "bpfile": "./station_jodc.bp",
}


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


def _station_codes_from_bp_meta(bp_meta: dict, wl_only: bool = True) -> list[str]:
    codes = []
    seen = set()
    for sid, meta in bp_meta.items():
        name = str(meta.get("name", "")).strip()
        parts = name.split()
        code = parts[0] if len(parts) > 0 else str(sid)
        svar = parts[1].upper() if len(parts) > 1 else "WL"
        if wl_only and svar != "WL":
            continue
        if code in seen:
            continue
        seen.add(code)
        codes.append(code)
    return codes


def _unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _merge_config(args: argparse.Namespace) -> dict:
    cfg = dict(USER_CONFIG) if USER_CONFIG.get("enable", False) else {}

    if args.base_dir is not None:
        cfg["base_dir"] = args.base_dir
    if args.station is not None:
        cfg["station"] = args.station
    if args.stations is not None:
        cfg["stations"] = list(args.stations)
    if args.all is not None:
        cfg["all"] = bool(args.all)
    if args.use_bp_stations is not None:
        cfg["use_bp_stations"] = bool(args.use_bp_stations)
    if args.bp_wl_only is not None:
        cfg["bp_wl_only"] = bool(args.bp_wl_only)
    if args.start is not None:
        cfg["start"] = args.start
    if args.end is not None:
        cfg["end"] = args.end
    if args.tz_offset is not None:
        cfg["tz_offset"] = float(args.tz_offset)
    if args.scale is not None:
        cfg["scale"] = float(args.scale)
    if args.out is not None:
        cfg["out"] = args.out
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.bpfile is not None:
        cfg["bpfile"] = args.bpfile

    # Final defaults (match legacy behavior)
    cfg.setdefault("base_dir", None)
    cfg.setdefault("station", None)
    cfg.setdefault("stations", None)
    cfg.setdefault("all", False)
    cfg.setdefault("use_bp_stations", False)
    cfg.setdefault("bp_wl_only", True)
    cfg.setdefault("start", None)
    cfg.setdefault("end", None)
    cfg.setdefault("tz_offset", 0.0)
    cfg.setdefault("scale", 1.0)
    cfg.setdefault("out", None)
    cfg.setdefault("outdir", ".")
    cfg.setdefault("bpfile", None)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Convert JODC tide data to NPZ.")
    parser.add_argument("--base-dir", default=None, help="Base directory that contains station folders.")
    parser.add_argument("--station", default=None, help="Single station code (e.g., 0112, 2003, MA11)")
    parser.add_argument("--stations", nargs="+", default=None, help="Multiple station codes.")
    parser.add_argument("--all", dest="all", action="store_true", help="Process all station folders")
    parser.add_argument("--no-all", dest="all", action="store_false", help="Do not process all station folders")
    parser.set_defaults(all=None)
    parser.add_argument(
        "--stations-from-bpfile",
        dest="use_bp_stations",
        action="store_true",
        help="Select stations from bpfile comments and process only those.",
    )
    parser.add_argument(
        "--no-stations-from-bpfile",
        dest="use_bp_stations",
        action="store_false",
        help="Do not select stations from bpfile.",
    )
    parser.set_defaults(use_bp_stations=None)
    parser.add_argument(
        "--bp-wl-only",
        dest="bp_wl_only",
        action="store_true",
        help="When using bp stations, keep only entries tagged WL.",
    )
    parser.add_argument(
        "--no-bp-wl-only",
        dest="bp_wl_only",
        action="store_false",
        help="When using bp stations, use all bp entries.",
    )
    parser.set_defaults(bp_wl_only=None)
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--tz_offset", type=float, default=None, help="Hours to shift timestamps (local -> UTC)")
    parser.add_argument("--scale", type=float, default=None, help="Scale factor (e.g., 0.01 for cm->m)")
    parser.add_argument("--out", default=None, help="Output NPZ filename")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--bpfile", default=None, help="Optional station_jodc.bp file path")
    args = parser.parse_args()

    cfg = _merge_config(args)
    base = str(Path(cfg["base_dir"]).expanduser().resolve()) if cfg.get("base_dir") else os.path.dirname(os.path.abspath(__file__))

    bp_path = str(Path(cfg["bpfile"]).expanduser().resolve()) if cfg.get("bpfile") else os.path.join(base, "station_jodc.bp")
    bp_meta = _read_bp(bp_path)

    if cfg.get("stations"):
        stations = [str(s).strip() for s in cfg["stations"] if str(s).strip()]
    elif cfg.get("station"):
        stations = [str(cfg["station"]).strip()]
    elif bool(cfg.get("use_bp_stations", False)):
        stations = _station_codes_from_bp_meta(bp_meta, wl_only=bool(cfg.get("bp_wl_only", True)))
        if len(stations) == 0:
            raise SystemExit(f"No stations selected from bpfile: {bp_path}")
    elif bool(cfg.get("all", False)):
        stations = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    else:
        raise SystemExit("Provide stations via config/CLI (stations/station) or set all=true/--all.")
    stations = _unique_preserve_order(stations)
    print(
        f"[INFO] station selection: n={len(stations)}, "
        f"source={'bpfile' if bool(cfg.get('use_bp_stations', False)) else 'manual/all'}",
        flush=True,
    )

    records_time = []
    records_value = []
    records_station = []
    code_to_id = {}
    id_to_code = {}
    for sid, meta in bp_meta.items():
        name = meta.get("name", "")
        code = name.split()[0] if name else None
        if code:
            code_to_id[code] = int(sid)
            id_to_code[int(sid)] = code
    used_ids = set(code_to_id.values())
    next_fallback_id = (max(used_ids) + 1) if len(used_ids) > 0 else 1
    station_id_map = {}
    station_names = []

    for idx, code in enumerate(sorted(stations), start=1):
        folder = os.path.join(base, code)
        if not os.path.isdir(folder):
            print(f"[WARN] station folder not found, skip: {folder}", flush=True)
            continue
        station_code, times, values = _load_station(
            folder,
            cfg.get("start"),
            cfg.get("end"),
            float(cfg.get("tz_offset", 0.0)),
            float(cfg.get("scale", 1.0)),
        )
        if times.size == 0:
            continue
        # Match station code to bp meta; fallback to unique IDs above max(bp_id)
        sid = code_to_id.get(code)
        if sid is None:
            sid = next_fallback_id
            while sid in used_ids:
                sid += 1
            next_fallback_id = sid + 1
            print(
                f"[WARN] station code '{code}' not found in bp metadata; assigned fallback id={sid}",
                flush=True,
            )
        used_ids.add(int(sid))
        station_id_map[code] = sid
        station_names.append(code)
        tnum = date2num(list(times))
        records_time.extend(tnum)
        records_value.extend(values.tolist())
        records_station.extend([station_id_map[code]] * len(values))

    # Safety check: each station code must map to a unique station id.
    sid_vals = list(station_id_map.values())
    if len(set(sid_vals)) != len(sid_vals):
        raise RuntimeError(
            "Station ID collision detected in station_id_map. "
            "Please verify bp metadata and station folder naming."
        )

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

    os.makedirs(str(cfg["outdir"]), exist_ok=True)
    if cfg.get("out"):
        out_path = os.path.join(str(cfg["outdir"]), str(cfg["out"]))
    else:
        tag = "all" if bool(cfg.get("all", False)) and not cfg.get("station") and not cfg.get("stations") else station_names[0]
        out_path = os.path.join(str(cfg["outdir"]), f"jodc_tide_{tag}.npz")
    savez(out_path, bundle)
    print(f"[OK] wrote NPZ: {out_path}")


if __name__ == "__main__":
    main()
