#!/usr/bin/env python3
"""
Process Onagawa Bay D1 CTD xlsx files (edit CONFIG below):
- read CTD profiles from multiple sheets (all bays)
- map station coordinates from "緯度経度" sheet when available
- plot selected variables by station/date
- optionally save npz for downstream use
- optionally write station.in for SCHISM extraction

Examples:
- Plot with CONFIG defaults:
  python process_onagawa_d1_ctd.py
- Save npz:
  python process_onagawa_d1_ctd.py --save-npz
"""
import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting
    plt = None


CONFIG = {  # Edit this block to change defaults.
    "BASE_DIR": "/Users/kpark/Codes/D26-017-selected/OnagawaBayData/D1",
    "CTD_FILES": [
        "0003_D01/FY2012_Rinko-Profilerat5oceanareasinMiyagiPrefecture.xlsx",
        "0061_D26/FY2013_Rinko-Profilerat3oceanareasinMiyagiPrefecture.xlsx",
        "0061_D01/0061_D01/1-3 CTD Ongawa Bay.xlsx",
    ],
    "PLOT_DIR": "/Users/kpark/Codes/D26-017-selected/OnagawaD1Plots",
    "PLOT_VARS": ["temp", "sal", "chl_flu", "chl_a"],
    "SAVE_VARS": [
        "temp",
        "sal",
        "cond",
        "ec25",
        "density",
        "sigma_t",
        "chl_flu",
        "chl_a",
        "turbidity",
        "do_pct",
        "do_mg",
    ],
    "MAX_PROFILES": None,
    "SAVE_NPZ": True,
    "NPZ_PATH": "/Users/kpark/Codes/D26-017-selected/npz/onagawa_d1_ctd.npz",
    "LOCAL_TIME_UTC_OFFSET_HOURS": 0.0,
    "PLOT_TIME_UTC": True,
    "SKIP_PLOT": False,
    "SKIP_STATION_IDS": {"地点", "チテン"},
    "STATION_LABEL_STYLE": "bay_station",  # station, bay_station, dataset_station
    "BAY_NAME_MAP": {
        "女川湾": "Onagawa Bay",
        "雄勝湾": "Ogatsu Bay",
        "長面浦": "Nagatsura Lagoon",
        "仙台湾": "Sendai Bay",
        "気仙沼湾": "Kesennuma Bay",
        "志津川湾": "Shizugawa Bay",
        "データ": "Onagawa Bay"
    },
    "WRITE_STATION_IN": True,
    "STATION_IN_PATH": "./station_onagawa_d1.in",
    "STATION_IN_FLAGS": "1 0 0 0 1 1 1 1 1",
    "STATION_IN_DEPTH": 0.0,
    "LON_WRAP_360": False,
}

COL_IDX = {
    "year": 0,
    "date": 1,
    "station": 2,
    "depth": 3,
    "temp": 4,
    "sal": 5,
    "cond": 6,
    "ec25": 7,
    "density": 8,
    "sigma_t": 9,
    "chl_flu": 10,
    "chl_a": 11,
    "turbidity": 12,
    "do_pct": 13,
    "do_mg": 14,
}

VAR_LABELS = {
    "temp": "Temperature (C)",
    "sal": "Salinity",
    "cond": "Conductivity (mS/cm)",
    "ec25": "EC25 (uS/cm)",
    "density": "Density (kg/m3)",
    "sigma_t": "Sigma-T",
    "chl_flu": "Chl-Flu (ppb)",
    "chl_a": "Chl-a (ug/L)",
    "turbidity": "Turbidity",
    "do_pct": "DO (%)",
    "do_mg": "DO (mg/L)",
}


def _wrap_lon(lon, wrap_360):
    if lon is None:
        return None
    if not wrap_360:
        return lon
    if lon < 0:
        return lon + 360.0
    if lon <= 180.0:
        return lon + 360.0
    return lon


def _shift_time(dt, hours):
    if dt is None or hours is None:
        return dt
    return dt + timedelta(hours=hours)


def _local_to_utc(dt, offset_hours):
    return _shift_time(dt, -offset_hours) if offset_hours is not None else dt


def _plot_time(dt, use_utc):
    if not use_utc:
        return dt
    offset_hours = CONFIG.get("LOCAL_TIME_UTC_OFFSET_HOURS", 0.0)
    return _local_to_utc(dt, offset_hours)


def _station_label(dataset, bay, station_id, style):
    station_id = station_id or ""
    if style == "station":
        return station_id or dataset
    if style == "bay_station":
        if station_id:
            return f"{bay}_{station_id}"
        return bay or dataset
    if style == "dataset_station":
        if station_id:
            return f"{dataset}_{station_id}"
        return dataset
    return station_id or dataset


def _cell_str(val):
    if val is None:
        return ""
    s = str(val).strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff")
    return s


def _to_float(val):
    try:
        return float(str(val))
    except Exception:
        return None


def _clean_station_id(val):
    s = _cell_str(val)
    if not s:
        return ""
    f = _to_float(s)
    if f is not None and abs(f - int(f)) < 1e-6:
        return str(int(f))
    return s


def _is_year(val):
    f = _to_float(val)
    return f is not None and 1900 <= f <= 2100


def _parse_excel_serial(val):
    try:
        f = float(val)
    except Exception:
        return None
    if f < 30000 or f > 60000:
        return None
    base = datetime(1899, 12, 30)
    return base + timedelta(days=f)


def _parse_date(val):
    d = _parse_excel_serial(val)
    if d is not None:
        return d
    s = _cell_str(val)
    if not s:
        return None
    s = s.replace(".", "/").replace("-", "/")
    if "/" in s:
        parts = s.split("/")
        if len(parts) >= 3 and all(p.strip().isdigit() for p in parts[:3]):
            y = int(parts[0])
            if y < 100:
                y += 2000
            m = int(parts[1])
            d = int(parts[2])
            try:
                return datetime(y, m, d)
            except Exception:
                return None
    if s.isdigit():
        if len(s) == 8:
            return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        if len(s) == 6:
            return datetime(int(s[0:2]) + 2000, int(s[2:4]), int(s[4:6]))
    return None


def xlsx_shared_strings(zf):
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    return [t.text or "" for t in root.iter(f"{ns}t")]


def xlsx_sheet_map(zf):
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rid_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {r.get("Id"): r.get("Target") for r in rels}
    sheets = []
    for s in wb.iter(f"{ns}sheet"):
        name = s.get("name")
        rid = s.get(f"{rid_ns}id")
        target = rel_map.get(rid, "")
        if target.startswith("/"):
            target = target[1:]
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((name, target))
    return sheets


def read_sheet_rows(zf, sheet_path, shared_strings, max_rows=None):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = ET.fromstring(zf.read(sheet_path))
    rows = []
    for row in root.iter(f"{ns}row"):
        cells = []
        for c in row.iter(f"{ns}c"):
            r = c.get("r")
            if r:
                m = re.match(r"[A-Z]+", r)
                idx = 0
                for ch in m.group(0):
                    idx = idx * 26 + (ord(ch) - 64)
                idx -= 1
            else:
                idx = len(cells)
            while len(cells) <= idx:
                cells.append("")
            v = c.find(f"{ns}v")
            if c.get("t") == "s" and v is not None:
                sval = shared_strings[int(v.text)] if v.text and v.text.isdigit() else ""
                cells[idx] = sval
            elif c.get("t") == "inlineStr":
                t = c.find(f"{ns}is/{ns}t")
                cells[idx] = t.text if t is not None else ""
            else:
                cells[idx] = v.text if v is not None else ""
        if cells:
            rows.append(cells)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def _is_lat(val):
    f = _to_float(val)
    return f is not None and 30.0 <= f <= 50.0


def _is_lon(val):
    f = _to_float(val)
    return f is not None and 120.0 <= f <= 160.0


def parse_latlon_sheet(rows):
    coords = {}
    for row in rows:
        lat = None
        lon = None
        for cell in row:
            if lat is None and _is_lat(cell):
                lat = _to_float(cell)
                continue
            if lon is None and _is_lon(cell):
                lon = _to_float(cell)
        if lat is None or lon is None:
            continue
        sid = None
        for cell in row:
            s = _cell_str(cell)
            if not s:
                continue
            if _is_lat(s) or _is_lon(s):
                continue
            sid = _clean_station_id(s)
            break
        if not sid:
            continue
        coords[sid] = (lat, lon)
    return coords


def parse_ctd_workbook(path):
    records = []
    try:
        with zipfile.ZipFile(path) as zf:
            shared = xlsx_shared_strings(zf)
            sheets = xlsx_sheet_map(zf)
            latlon_rows = []
            for name, sheet in sheets:
                if "緯度" in name and "経度" in name:
                    latlon_rows = read_sheet_rows(zf, sheet, shared)
                    break
            coords = parse_latlon_sheet(latlon_rows) if latlon_rows else {}
            for name, sheet in sheets:
                if "緯度" in name and "経度" in name:
                    continue
                bay_name = CONFIG["BAY_NAME_MAP"].get(name, name)
                rows = read_sheet_rows(zf, sheet, shared)
                if not rows:
                    continue
                for row in rows:
                    if not row or len(row) < COL_IDX["depth"] + 1:
                        continue
                    if not _is_year(row[COL_IDX["year"]]):
                        continue
                    row = list(row)
                    if len(row) < 15:
                        row.extend([""] * (15 - len(row)))
                    date_val = _parse_date(row[COL_IDX["date"]])
                    if date_val is None:
                        continue
                    station_id = _clean_station_id(row[COL_IDX["station"]])
                    if not station_id or station_id in CONFIG["SKIP_STATION_IDS"]:
                        continue
                    depth = _to_float(row[COL_IDX["depth"]])
                    if depth is None:
                        continue
                    lat = None
                    lon = None
                    if station_id in coords:
                        lat, lon = coords[station_id]
                    rec = {
                        "time": date_val,
                        "station_id": station_id,
                        "station_name": station_id,
                        "bay": bay_name,
                        "lat": lat,
                        "lon": lon,
                        "depth": depth,
                    }
                    for key in CONFIG["SAVE_VARS"]:
                        idx = COL_IDX.get(key)
                        if idx is None or idx >= len(row):
                            rec[key] = None
                        else:
                            rec[key] = _to_float(row[idx])
                    records.append(rec)
    except zipfile.BadZipFile:
        return []
    return records


def collect_records(base_dir):
    records = []
    files = CONFIG["CTD_FILES"]
    if files:
        paths = [os.path.join(base_dir, p) for p in files]
    else:
        paths = []
        for root, _, names in os.walk(base_dir):
            for name in names:
                if name.startswith("~$") or name.startswith("._") or not name.lower().endswith(".xlsx"):
                    continue
                paths.append(os.path.join(root, name))
    for path in paths:
        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue
        rel = os.path.relpath(path, base_dir)
        dataset = rel.split(os.sep)[0]
        records_path = parse_ctd_workbook(path)
        for rec in records_path:
            rec["dataset"] = dataset
            rec["source"] = os.path.basename(path)
            rec["station_label"] = _station_label(
                dataset,
                rec.get("bay", ""),
                rec.get("station_id", ""),
                CONFIG["STATION_LABEL_STYLE"],
            )
        records.extend(records_path)
    return records


def plot_profiles(records, plot_vars, out_dir, max_profiles=None):
    if plt is None:
        raise RuntimeError("matplotlib is not available for plotting")
    if not plot_vars:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    use_utc = CONFIG.get("PLOT_TIME_UTC", False)
    tz_label = "UTC" if use_utc else "local"
    groups = {}
    for rec in records:
        plot_dt = _plot_time(rec["time"], use_utc)
        key = (plot_dt.date(), rec.get("station_label") or rec.get("station_id"))
        groups.setdefault(key, []).append(rec)
    count = 0
    for (date, station), recs in sorted(groups.items()):
        if max_profiles is not None and count >= max_profiles:
            break
        plot_keys = []
        for key in plot_vars:
            vals = [r.get(key) for r in recs if r.get(key) is not None and r.get("depth") is not None]
            if vals:
                plot_keys.append(key)
        if not plot_keys:
            continue
        fig, axes = plt.subplots(1, len(plot_keys), figsize=(4 * len(plot_keys), 6), sharey=True)
        if len(plot_keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, plot_keys):
            pairs = [(r["depth"], r.get(key)) for r in recs if r.get(key) is not None and r.get("depth") is not None]
            pairs.sort(key=lambda x: x[0])
            depths = [p[0] for p in pairs]
            vals = [p[1] for p in pairs]
            ax.plot(vals, depths, "-o", ms=3)
            ax.set_xlabel(VAR_LABELS.get(key, key))
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Depth (m)")
        axes[0].invert_yaxis()
        title = f"{station} {date.isoformat()} ({tz_label})"
        fig.suptitle(title)
        fig.tight_layout()
        out_name = f"ctd_{date.strftime('%Y%m%d')}_{station}.png".replace(" ", "_")
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        count += 1
    return count


def save_npz(records, out_path, save_vars):
    if np is None:
        raise RuntimeError("numpy is required to save npz output")
    offset_hours = CONFIG.get("LOCAL_TIME_UTC_OFFSET_HOURS", 0.0)
    time_local = [r["time"] for r in records]
    time_utc = [_local_to_utc(t, offset_hours) for t in time_local]
    payload = {
        "time": np.array(time_utc, dtype="datetime64[s]"),
        "time_local": np.array(time_local, dtype="datetime64[s]"),
        "station_id": np.array([r.get("station_id", "") for r in records], dtype="U"),
        "station_name": np.array([r.get("station_name", "") for r in records], dtype="U"),
        "lat": np.array([np.nan if r.get("lat") is None else r.get("lat") for r in records], dtype=float),
        "lon": np.array([np.nan if r.get("lon") is None else r.get("lon") for r in records], dtype=float),
        "depth": np.array([r.get("depth") for r in records], dtype=float),
        "source": np.array([r.get("source", "") for r in records], dtype="U"),
        "dataset": np.array([r.get("dataset", "") for r in records], dtype="U"),
        "bay": np.array([r.get("bay", "") for r in records], dtype="U"),
    }
    for key in save_vars:
        payload[key] = np.array(
            [np.nan if r.get(key) is None else r.get(key) for r in records],
            dtype=float,
        )
    np.savez(out_path, **payload)


def write_station_in(records, out_path, flags, depth, wrap_360):
    stations = []
    seen = set()
    for rec in records:
        lat = rec.get("lat")
        lon = rec.get("lon")
        if lat is None or lon is None:
            continue
        name = rec.get("station_label") or rec.get("station_id")
        lon_val = _wrap_lon(lon, wrap_360)
        key = (name, round(lat, 6), round(lon_val, 6))
        if key in seen:
            continue
        seen.add(key)
        stations.append((name, lon_val, lat))
    if not stations:
        return 0
    stations.sort(key=lambda x: x[0])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{flags}\n")
        f.write(f"{len(stations)}\n")
        for idx, (name, lon, lat) in enumerate(stations, start=1):
            f.write(f"{idx} {lon:.6f} {lat:.6f} {depth:.6f} # {name}\n")
    return len(stations)


def main():
    parser = argparse.ArgumentParser(description="Process Onagawa D1 CTD xlsx files.")
    parser.add_argument("--base", default=CONFIG["BASE_DIR"], help="Base D1 directory")
    parser.add_argument("--plot-dir", default=None, help="Output plot directory")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--save-npz", action="store_true", help="Save npz output")
    parser.add_argument("--npz-path", default=None, help="Path to write npz")
    parser.add_argument("--max-profiles", type=int, default=None, help="Limit number of plots")
    args = parser.parse_args()

    base_dir = args.base or CONFIG["BASE_DIR"]
    plot_vars = CONFIG["PLOT_VARS"]
    save_vars = CONFIG["SAVE_VARS"]
    skip_plot = CONFIG["SKIP_PLOT"] or args.no_plot
    max_profiles = args.max_profiles if args.max_profiles is not None else CONFIG["MAX_PROFILES"]
    plot_dir = args.plot_dir or CONFIG["PLOT_DIR"] or os.path.join(base_dir, "plots_ctd")
    save_npz_flag = args.save_npz or CONFIG["SAVE_NPZ"]
    npz_path = args.npz_path or CONFIG["NPZ_PATH"] or os.path.join(base_dir, "onagawa_d1_ctd.npz")
    write_station_in_flag = CONFIG["WRITE_STATION_IN"]
    station_in_path = CONFIG["STATION_IN_PATH"] or os.path.join(base_dir, "station.in")
    station_flags = CONFIG["STATION_IN_FLAGS"]
    station_depth = CONFIG["STATION_IN_DEPTH"]
    wrap_360 = CONFIG["LON_WRAP_360"]

    records = collect_records(base_dir)
    if not records:
        print("No CTD records found.")
        return 1
    print(f"Loaded {len(records)} CTD samples from {base_dir}")

    if not skip_plot:
        count = plot_profiles(records, plot_vars, plot_dir, max_profiles=max_profiles)
        print(f"Saved {count} profile plots to {plot_dir}")

    if save_npz_flag:
        if np is None:
            print("numpy is required for --save-npz")
            return 1
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        save_npz(records, npz_path, save_vars)
        print(f"Saved npz to {npz_path}")
    if write_station_in_flag:
        count = write_station_in(records, station_in_path, station_flags, station_depth, wrap_360)
        print(f"Saved station.in with {count} stations to {station_in_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
