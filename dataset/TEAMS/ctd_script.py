"""
CTD Data Viewer & Plotter for ODV4 (.o4x), .dat, and .csv formats.

Usage examples:
  python ctd_script.py --file MR12-E02_Leg2_CTD.o4x
  python ctd_script.py --file ./KH-18-J02C_ctd_dmo
  python ctd_script.py --base /Users/kpark/Codes/coastal-ocean-utils/dataset/TEAMS --list
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    print(f"Reading file: {filename}...")
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    variables = []
    stations = []

    reading_vars = False
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        if clean_line.startswith("#"):
            break
        if clean_line.startswith("Variables:"):
            reading_vars = True
            continue
        if reading_vars:
            match = re.match(r"^(.*?)\s+[\d\.]+$", clean_line)
            if match:
                variables.append(match.group(1).strip())
            else:
                parts = clean_line.split()
                if len(parts) > 1:
                    variables.append(" ".join(parts[:-1]))

    current_meta = None
    data_buffer = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if current_meta and data_buffer:
                stations.append({"meta": current_meta, "data": data_buffer})

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
                "BottomDepth": parts[lon_idx + 2] if len(parts) > lon_idx + 2 else "N/A",
                "Source": os.path.basename(filename),
            }
            data_buffer = []
        elif not any(x in line for x in ["ODV", "File", "Variables", "Type", "Nstat"]):
            if variables and variables[0].split()[0] in line:
                continue
            try:
                row = [float(x) for x in line.split()]
                if len(row) >= len(variables) * 2:
                    values = row[0::2]
                    data_buffer.append(values)
            except ValueError:
                continue

    if current_meta and data_buffer:
        stations.append({"meta": current_meta, "data": data_buffer})

    processed = []
    for s in stations:
        cols = variables[: len(s["data"][0])] if s["data"] else []
        df = pd.DataFrame(s["data"], columns=cols)
        processed.append({"meta": s["meta"], "df": df})
    return processed


def parse_dat_file(filename):
    print(f"Reading file: {filename}...")
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header = None
    data_rows = []
    for line in lines:
        if line.startswith("#") and header is None:
            header = line[1:].strip()
            continue
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        if len(nums) < 2:
            continue
        data_rows.append([float(x) for x in nums])

    if not data_rows:
        return []

    meta = {
        "Cruise": "N/A",
        "Station": os.path.splitext(os.path.basename(filename))[0],
        "Date": "N/A",
        "Time": "N/A",
        "Longitude": 0.0,
        "Latitude": 0.0,
        "BottomDepth": "N/A",
        "Source": os.path.basename(filename),
    }

    if header:
        tokens = header.split()
        date_idx = next((i for i, t in enumerate(tokens) if re.match(r"^\d{8}$", t)), None)
        if date_idx is not None:
            meta["Date"] = tokens[date_idx]
            if date_idx + 1 < len(tokens) and re.match(r"^\d{4}$", tokens[date_idx + 1]):
                meta["Time"] = tokens[date_idx + 1]
        if len(tokens) > 1:
            meta["Cruise"] = tokens[1]
        if date_idx is not None and date_idx > 2:
            meta["Station"] = tokens[2]
        lat_tok = next((t for t in tokens if re.search(r"[NS]$", t)), None)
        lon_tok = next((t for t in tokens if re.search(r"[EW]$", t)), None)
        if lat_tok:
            meta["Latitude"] = _parse_dms(lat_tok)
        if lon_tok:
            meta["Longitude"] = _parse_dms(lon_tok)
        if tokens:
            for t in reversed(tokens):
                if re.match(r"^\d+(?:\.\d+)?$", t):
                    meta["BottomDepth"] = t
                    break

    ncol = max(len(r) for r in data_rows)
    cols = ["Depth [m]", "Temperature [degC]", "Salinity [psu]"]
    if ncol > len(cols):
        cols += [f"Var{i}" for i in range(4, ncol + 1)]
    cols = cols[:ncol]
    df = pd.DataFrame([r[:ncol] for r in data_rows], columns=cols)

    return [{"meta": meta, "df": df}]


def parse_csv_file(filename):
    print(f"Reading file: {filename}...")
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
    bot_col = _pick_column(df.columns, ["bottom", "bot"], exclude=["bottle"])

    group_cols = [c for c in [cruise_col, station_col, date_col, time_col] if c]
    if not group_cols:
        group_cols = [station_col] if station_col else []

    stations = []
    if group_cols:
        groups = df.groupby(group_cols, dropna=False)
    else:
        groups = [(None, df)]

    for key, g in groups:
        meta = {
            "Cruise": str(g[cruise_col].iloc[0]) if cruise_col else "N/A",
            "Station": str(g[station_col].iloc[0]) if station_col else "N/A",
            "Date": str(g[date_col].iloc[0]) if date_col else "N/A",
            "Time": str(g[time_col].iloc[0]) if time_col else "N/A",
            "Longitude": float(g[lon_col].iloc[0]) if lon_col else 0.0,
            "Latitude": float(g[lat_col].iloc[0]) if lat_col else 0.0,
            "BottomDepth": str(g[bot_col].iloc[0]) if bot_col else "N/A",
            "Source": os.path.basename(filename),
        }
        stations.append({"meta": meta, "df": g.reset_index(drop=True)})

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
        else:
            print(f"Skipping unsupported file: {path}")
    return stations


def print_summary(stations):
    if not stations:
        print("No stations found.")
        return

    header = (
        f"{'Station':<10} {'Date':<12} {'Time':<8} {'Lat (N)':<10} "
        f"{'Lon (E)':<10} {'Bot.Depth':<12} {'MaxObsDepth':<15} {'Source':<20}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for s in stations:
        m = s["meta"]
        df = s["df"]
        depth_col = _pick_column(df.columns, ["depth"], exclude=["bottom", "bot"])
        if depth_col is None:
            depth_col = _pick_column(df.columns, ["pressure"])
        max_depth = df[depth_col].max() if depth_col else 0
        print(
            f"{m['Station']:<10} {m['Date']:<12} {m['Time']:<8} "
            f"{float(m['Latitude']):<10.4f} {float(m['Longitude']):<10.4f} "
            f"{m['BottomDepth']:<12} {max_depth:<15.1f} {m['Source']:<20}"
        )

    print(sep)
    print(f"Total Stations: {len(stations)}\n")


def plot_data(stations, save_to_file=False, max_stations=None):
    if not stations:
        return

    if max_stations is not None:
        stations = stations[:max_stations]

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    for s in stations:
        df = s["df"]
        depth_col = _pick_column(df.columns, ["depth"], exclude=["bottom", "bot"])
        if depth_col is None:
            depth_col = _pick_column(df.columns, ["pressure"])
        temp_col = _pick_column(df.columns, ["temperature", "temp"], exclude=["potential"])
        sal_col = _pick_column(df.columns, ["salinity", "psu"])

        if not depth_col:
            continue
        label = f"{s['meta']['Station']}"

        if temp_col and temp_col in df:
            axes[0].plot(df[temp_col], df[depth_col], label=label)
        if sal_col and sal_col in df:
            axes[1].plot(df[sal_col], df[depth_col], label=label)

    axes[0].set_title("Temperature Profile")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Depth")
    axes[0].grid(True, alpha=0.5, linestyle="--")
    axes[0].invert_yaxis()

    axes[1].set_title("Salinity Profile")
    axes[1].set_xlabel("Salinity")
    axes[1].grid(True, alpha=0.5, linestyle="--")

    axes[0].legend(loc="best", fontsize="x-small", title="Stations")
    plt.tight_layout()

    if save_to_file:
        plt.savefig("ctd_profile.png", dpi=300)
        print("Plot saved to ctd_profile.png")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTD Data Viewer")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="File, directory, or glob pattern to parse.",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Base directory to scan for CTD folders (name contains 'ctd').",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List detected CTD files and exit.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save plot as image instead of showing it.",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=None,
        help="Limit number of stations plotted.",
    )
    args = parser.parse_args()

    files = _collect_files(args.file, args.base)
    if not files:
        print("No CTD files found. Use --file or --base.")
        sys.exit(1)

    if args.list:
        print("CTD files:")
        for f in files:
            print(f"  {f}")
        sys.exit(0)

    stations = load_stations(files)
    print_summary(stations)
    plot_data(stations, save_to_file=args.save, max_stations=args.max_stations)
