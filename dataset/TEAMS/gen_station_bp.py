#!/usr/bin/env python3
"""
Generate station.bp from CTD/XCTD data files.

Usage examples:
  python gen_station_bp.py --base /Users/kpark/Codes/coastal-ocean-utils/dataset/TEAMS \
    --output station_ctd.bp
  python gen_station_bp.py --file ./MR12-E02_leg2_ctd_pi/*.o4x --output station_ctd.bp
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


SUPPORTED_EXTS = {".o4x", ".dat", ".csv"}


def _parse_dms(token):
    token = token.strip()
    m = re.match(r"^(\d+)-(\d+(?:\.\d+)?)([NSEW])$", token)
    if not m:
        try:
            return float(token)
        except ValueError:
            return 0.0
    deg = float(m.group(1))
    minutes = float(m.group(2))
    hemi = m.group(3)
    val = deg + minutes / 60.0
    if hemi in ["S", "W"]:
        val = -val
    return val


def _pick_column(cols, keywords, exclude=()):
    for col in cols:
        low = col.lower()
        if any(k in low for k in keywords) and not any(e in low for e in exclude):
            return col
    return None


def _normalize_columns(df):
    cols = []
    for c in df.columns:
        c = str(c).replace("\u3000", " ")
        c = c.replace("°", "deg")
        cols.append(c.strip())
    df.columns = cols
    return df


def _collect_files(file_arg=None, base=None):
    files = []
    if base:
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Base directory not found: {base}")
        for name in sorted(os.listdir(base)):
            if "ctd" not in name.lower():
                continue
            dpath = os.path.join(base, name)
            if not os.path.isdir(dpath):
                continue
            for root, _, fnames in os.walk(dpath):
                for fname in fnames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in SUPPORTED_EXTS:
                        files.append(os.path.join(root, fname))
    else:
        if not file_arg:
            return []
        candidates = glob.glob(file_arg)
        if not candidates and os.path.exists(file_arg):
            candidates = [file_arg]
        for path in candidates:
            if os.path.isdir(path):
                for root, _, fnames in os.walk(path):
                    for fname in fnames:
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in SUPPORTED_EXTS:
                            files.append(os.path.join(root, fname))
            else:
                ext = os.path.splitext(path)[1].lower()
                if ext in SUPPORTED_EXTS:
                    files.append(path)
    return sorted(set(files))


def parse_odv_file(filename):
    stations = []
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_meta = None
    data_seen = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_meta and data_seen:
                stations.append(current_meta)
            parts = line[1:].strip().split()
            time_str = "N/A"
            lon_idx = 4
            if len(parts) > 4 and ":" in parts[4]:
                time_str = parts[4]
                lon_idx = 5
            current_meta = {
                "Cruise": parts[0] if len(parts) > 0 else "N/A",
                "Station": parts[1] if len(parts) > 1 else "N/A",
                "Date": parts[3] if len(parts) > 3 else "N/A",
                "Time": time_str,
                "Longitude": float(parts[lon_idx]) if len(parts) > lon_idx else 0.0,
                "Latitude": float(parts[lon_idx + 1]) if len(parts) > lon_idx + 1 else 0.0,
                "Source": os.path.basename(filename),
            }
            data_seen = False
        elif not any(x in line for x in ["ODV", "File", "Variables", "Type", "Nstat"]):
            try:
                _ = [float(x) for x in line.split()]
                data_seen = True
            except ValueError:
                continue

    if current_meta and data_seen:
        stations.append(current_meta)

    return stations


def parse_dat_file(filename):
    stations = []
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header = next((ln.strip() for ln in lines if ln.startswith("#")), "")
    meta = {
        "Cruise": "N/A",
        "Station": os.path.splitext(os.path.basename(filename))[0],
        "Date": "N/A",
        "Time": "N/A",
        "Longitude": 0.0,
        "Latitude": 0.0,
        "Source": os.path.basename(filename),
    }

    if header:
        tokens = header[1:].strip().split()
        if tokens:
            if tokens[0] in ["CTD", "XCTD"]:
                meta["Cruise"] = tokens[1] if len(tokens) > 1 else "N/A"
                date_idx = next((i for i, t in enumerate(tokens) if re.match(r"^\d{8}$", t)), None)
                if date_idx is not None:
                    meta["Date"] = tokens[date_idx]
                    if date_idx + 1 < len(tokens) and re.match(r"^\d{4}$", tokens[date_idx + 1]):
                        meta["Time"] = tokens[date_idx + 1]
                if len(tokens) > 2 and not re.match(r"^\d{8}$", tokens[2]):
                    meta["Station"] = tokens[2]
            else:
                meta["Cruise"] = tokens[0]

            lat_tok = next((t for t in tokens if re.search(r"[NS]$", t)), None)
            lon_tok = next((t for t in tokens if re.search(r"[EW]$", t)), None)
            if lat_tok:
                meta["Latitude"] = _parse_dms(lat_tok)
            if lon_tok:
                meta["Longitude"] = _parse_dms(lon_tok)

    stations.append(meta)
    return stations


def parse_csv_file(filename):
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Failed to read CSV: {filename} ({e})")
        return []

    df = _normalize_columns(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    cruise_col = _pick_column(df.columns, ["cruise"])
    station_col = _pick_column(df.columns, ["station"])
    date_col = _pick_column(df.columns, ["mon/day/yr", "date"])
    time_col = _pick_column(df.columns, ["hh:mm", "time"], exclude=["zone"])
    lon_col = _pick_column(df.columns, ["lon"])
    lat_col = _pick_column(df.columns, ["lat"])

    group_cols = [c for c in [cruise_col, station_col, date_col, time_col] if c]
    if not group_cols:
        group_cols = [station_col] if station_col else []

    stations = []
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]

    for _, g in groups:
        meta = {
            "Cruise": str(g[cruise_col].iloc[0]) if cruise_col else "N/A",
            "Station": str(g[station_col].iloc[0]) if station_col else os.path.basename(filename),
            "Date": str(g[date_col].iloc[0]) if date_col else "N/A",
            "Time": str(g[time_col].iloc[0]) if time_col else "N/A",
            "Longitude": float(g[lon_col].iloc[0]) if lon_col else 0.0,
            "Latitude": float(g[lat_col].iloc[0]) if lat_col else 0.0,
            "Source": os.path.basename(filename),
        }
        stations.append(meta)

    return stations


def load_stations(files):
    stations = []
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".o4x":
            stations.extend(parse_odv_file(path))
        elif ext == ".dat":
            stations.extend(parse_dat_file(path))
        elif ext == ".csv":
            stations.extend(parse_csv_file(path))
    return stations


def _station_name(meta):
    cruise = str(meta.get("Cruise", "N/A")).strip()
    station = str(meta.get("Station", "N/A")).strip()
    if cruise not in ["", "N/A"] and station not in ["", "N/A"]:
        name = f"{cruise}_{station}"
    elif station not in ["", "N/A"]:
        name = station
    else:
        name = meta.get("Source", "station")
    return name.replace(" ", "_")


def write_station_bp(stations, output, round_digits=6):
    seen = set()
    rows = []
    for meta in stations:
        lon = float(meta.get("Longitude", 0.0))
        lat = float(meta.get("Latitude", 0.0))
        name = _station_name(meta)
        key = (name, round(lon, round_digits), round(lat, round_digits))
        if key in seen:
            continue
        seen.add(key)
        rows.append((lon, lat, name))

    with open(output, "w", encoding="utf-8") as f:
        f.write("1 1 1 1 1 1 1 1 1 !on (1)|off(0) flags for elev, air pressure, windx, windy, T, S, u, v, w\n")
        f.write(f"{len(rows)}\n")
        for i, (lon, lat, name) in enumerate(rows, start=1):
            f.write(f"{i:5d} {lon:11.6f} {lat:10.6f}     0 !{name}\n")

    print(f"Wrote {output} with {len(rows)} stations")


def main():
    parser = argparse.ArgumentParser(description="Generate station.bp from CTD data")
    parser.add_argument(
        "--base",
        default=None,
        help="Base directory to scan for CTD folders (name contains 'ctd').",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="File, directory, or glob pattern to parse.",
    )
    parser.add_argument(
        "--output",
        default="station_ctd.bp",
        help="Output station.bp file name.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=6,
        help="Decimal rounding for lon/lat deduplication (default: 6).",
    )
    args = parser.parse_args()

    files = _collect_files(args.file, args.base)
    if not files:
        print("No CTD files found. Use --base or --file.")
        return 1

    stations = load_stations(files)
    if not stations:
        print("No stations parsed from CTD files.")
        return 1

    write_station_bp(stations, args.output, round_digits=args.round)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
