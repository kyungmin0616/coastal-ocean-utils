#!/usr/bin/env python3
# download_uhslc_water_levels.py
#
# Download UHSLC water level time series for stations listed in a catalog CSV
# produced by build_uhslc_catalog2.py (or compatible). Supports selecting
# temporal resolution and a time window. Defaults to downloading the UHSLC
# WOCE text files for FAST daily/hourly using URLs already present in the
# catalog; optionally supports ERDDAP for RQDS if configured.

from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict
import io
import os
import sys
import time

import pandas as pd
import numpy as np
import requests
from pylib import zdata, savez, date2num

SECONDS_PER_DAY = 86400.0
EPOCH_NUM = date2num(pd.Timestamp("1970-01-01"))


# ==========================
# Editable Configuration
# ==========================
DEFAULT_CONFIG = {
    # Inputs
    "catalog": "uhslc_eastasia_catalog.csv",                  # path to catalog CSV (required via CLI or config)
    "out_dir": "uhslc_data",        # base output directory
    "resolution": "hourly_rqds",    # one of: hourly_fast, daily_fast, hourly_rqds
    "mode": "erddap",                 # 'woce' (uses URLs in catalog for FAST) or 'erddap'

    # Station selection
    "station_ids": None,             # list of station_ids to include (None = all)
    "max_stations": None,            # int limit for debugging

    # Time window (inclusive). Accepts YYYY-MM-DD or full ISO.
    "start": None,                   # e.g., "2015-01-01"
    "end": None,                     # e.g., "2020-12-31"

    # Networking
    "request_timeout": 60,
    "request_retries": 3,
    "request_sleep": 1.5,
    "max_workers": 4,               # parallel downloads

    # ERDDAP options (also used as fallback if WOCE parse fails)
    "erddap_base": "https://uhslc.soest.hawaii.edu/erddap/tabledap",
    # Mapping from resolution to ERDDAP dataset name
    "erddap_datasets": {
        "hourly_fast": "global_hourly_fast",
        "daily_fast": "global_daily_fast",
        "hourly_rqds": "global_hourly_rqds",
    },
    # Variables to request from ERDDAP (comma-separated). You may need to
    # adjust variable names to match ERDDAP. The default assumes a variable
    # named 'sea_level' exists. Change to 'ssh' or others if needed.
    "erddap_vars": "time,sea_level,uhslc_id",
    "erddap_fallback": True,         # if WOCE download/parse fails, try ERDDAP
    "value_columns": ["sea_level", "sea_level_adj", "ssh", "water_level", "surge"],

    # Output
    "write_csv": True,
    "write_combined": False,         # also write a combined CSV of all stations
    "write_npz": True,
    "npz_name": "uhslc_water_levels",
}


# ==========================
# Helpers
# ==========================

def _load_json_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    import json
    with open(os.fspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_template():
    import json
    print(json.dumps(DEFAULT_CONFIG, indent=2))


def _safe_to_datetime(x: Optional[str]) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        return None


def read_catalog(catalog_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(catalog_csv)
    # normalize expected columns
    want = {
        "station_id": "station_id",
        "lat": "lat",
        "lon": "lon",
        "fast_daily_url": "fast_daily_url",
        "fast_hourly_url": "fast_hourly_url",
        "authority_url": "authority_url",
        "source_network": "source_network",
        "station_name": "station_name",
        "country": "country",
    }
    colmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in want:
            colmap[c] = want[cl]
    df = df.rename(columns=colmap)
    if "station_id" not in df.columns:
        raise ValueError("Catalog must contain a 'station_id' column")
    df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")
    df = df.dropna(subset=["station_id"]).reset_index(drop=True)
    return df


def _fetch_text(url: str, *, tries: int, sleep: float, timeout: int) -> str:
    last = None
    for _ in range(int(tries)):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            r.encoding = r.encoding or "utf-8"
            return r.text
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise RuntimeError(f"Failed to download {url}: {last}")


def _fetch_csv(url: str, *, tries: int, sleep: float, timeout: int, **read_kwargs) -> pd.DataFrame:
    txt = _fetch_text(url, tries=tries, sleep=sleep, timeout=timeout)
    kwargs = {"low_memory": False}
    kwargs.update(read_kwargs)
    return pd.read_csv(io.StringIO(txt), **kwargs)


def parse_woce_daily(text: str) -> pd.DataFrame:
    """Parse WOCE daily text with flexible columns.

    Accepts lines with at least: year month day value [...]. Extra columns are ignored.
    """
    import re as _re
    times: List[pd.Timestamp] = []
    vals: List[float] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        toks = _re.split(r"\s+", s)
        nums: List[float] = []
        for t in toks:
            try:
                nums.append(float(t))
            except Exception:
                continue
        if len(nums) < 4:
            continue
        y, m, d = int(nums[0]), int(nums[1]), int(nums[2])
        val = float(nums[3])
        try:
            ts = pd.Timestamp(year=y, month=m, day=d, tz="UTC")
        except Exception:
            continue
        times.append(ts)
        vals.append(val)
    if not times:
        raise ValueError("No parsable daily rows found in WOCE text")
    return pd.DataFrame({"time": times, "value": vals})


def parse_woce_hourly(text: str) -> pd.DataFrame:
    """Parse WOCE hourly text with optional minute column.

    Accepts: year month day hour [minute] value [...]. Extra columns ignored.
    """
    import re as _re
    times: List[pd.Timestamp] = []
    vals: List[float] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        toks = _re.split(r"\s+", s)
        nums: List[float] = []
        for t in toks:
            try:
                nums.append(float(t))
            except Exception:
                continue
        if len(nums) < 5:
            continue
        y, m, d = int(nums[0]), int(nums[1]), int(nums[2])
        hour = int(nums[3])
        minute = 0
        # Heuristic: if next token looks like minute (0-59) and we have >= 6 tokens
        if len(nums) >= 6 and 0 <= int(nums[4]) <= 59:
            minute = int(nums[4])
            val = float(nums[5])
        else:
            val = float(nums[4])
        try:
            ts = pd.Timestamp(year=y, month=m, day=d, tz="UTC") + pd.Timedelta(hours=hour, minutes=minute)
        except Exception:
            continue
        times.append(ts)
        vals.append(val)
    if not times:
        raise ValueError("No parsable hourly rows found in WOCE text")
    return pd.DataFrame({"time": times, "value": vals})


def window_filter(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if start is not None:
        df = df[df["time"] >= start]
    if end is not None:
        df = df[df["time"] <= end]
    return df.reset_index(drop=True)


def build_erddap_url(base: str, dataset: str, varlist: str, station_id: int,
                     start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> str:
    # ERDDAP time constraints use ISO format with Z
    vars_q = varlist
    url = f"{base}/{dataset}.csv?{vars_q}&uhslc_id={int(station_id)}"
    if start is not None:
        url += f"&time>={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    if end is not None:
        url += f"&time<={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    return url


@dataclass
class Job:
    station_id: int
    url: str
    resolution: str
    mode: str
    station_name: str
    country: str
    lat: Optional[float] = None
    lon: Optional[float] = None


def _normalize_time(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    except (TypeError, ValueError):
        # Fallback to general parser
        return pd.to_datetime(series, utc=True, errors="coerce")


def _sanitize_component(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return "unknown"
    # Replace whitespace with underscores and remove problematic chars
    import re as _re
    s = _re.sub(r"\s+", "_", s)
    s = _re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return s or "unknown"


def _select_value_column(df: pd.DataFrame, tcol: str, candidates: Iterable[str]) -> str:
    # Prioritize explicit candidates (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns if c != tcol}
    for cand in candidates:
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]

    # Next choose first numeric-looking column excluding obvious ID columns
    for col in df.columns:
        if col == tcol:
            continue
        if "id" in col.lower():
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        if ser.notna().any():
            df[col] = ser
            return col

    # Fallback: first non-time column
    for col in df.columns:
        if col != tcol:
            return col
    raise ValueError("No value column available")


def _to_datenum(series: pd.Series) -> np.ndarray:
    tz = getattr(series.dt, "tz", None)
    values = series.dt.tz_convert(None) if tz is not None else series
    arr = values.to_numpy(dtype="datetime64[ns]", copy=False)
    ints = arr.astype("datetime64[ns]").astype("int64")
    mask = ints == np.iinfo(np.int64).min
    ints = ints.astype(np.float64)
    if mask.any():
        ints[mask] = np.nan
    days = ints / 1e9 / SECONDS_PER_DAY
    return days + EPOCH_NUM


def _build_bp(metadata: Dict[int, Dict[str, object]], station_ids: Iterable[int], resolutions: Optional[np.ndarray] = None) -> zdata:
    bp = zdata()
    station_ids_arr = np.asarray(list(station_ids), dtype=int)
    ordered = sorted({int(sid) for sid in station_ids_arr})

    names: List[str] = []
    lons: List[float] = []
    lats: List[float] = []
    countries: List[str] = []
    res_lookup: Dict[int, str] = {}
    if resolutions is not None:
        res_arr = np.asarray(list(resolutions))
        if res_arr.size == station_ids_arr.size:
            # Most common resolution per station
            for sid in ordered:
                mask = station_ids_arr == sid
                if mask.any():
                    vals = res_arr[mask]
                    if vals.size:
                        values, counts = np.unique(vals, return_counts=True)
                        res_lookup[sid] = values[int(np.argmax(counts))]
    res_values: List[str] = []
    for sid in ordered:
        meta = metadata.get(sid, {})
        name = meta.get("station_name") if isinstance(meta, dict) else None
        lon = meta.get("lon") if isinstance(meta, dict) else None
        lat = meta.get("lat") if isinstance(meta, dict) else None
        country = meta.get("country") if isinstance(meta, dict) else None
        default_res = meta.get("resolution") if isinstance(meta, dict) else None
        names.append(str(name) if name else str(sid))
        lons.append(float(lon) if lon is not None and not pd.isna(lon) else np.nan)
        lats.append(float(lat) if lat is not None and not pd.isna(lat) else np.nan)
        countries.append(str(country) if country else "")
        res_val = res_lookup.get(sid)
        if res_val is None and default_res:
            res_val = str(default_res)
        res_values.append(res_val or "")

    bp.station_id = np.array(ordered, dtype=int) if ordered else np.array([], dtype=int)
    bp.station_name = np.array(names, dtype="U128") if names else np.array([], dtype="U1")
    bp.lon = np.array(lons, dtype=float) if lons else np.array([], dtype=float)
    bp.lat = np.array(lats, dtype=float) if lats else np.array([], dtype=float)
    bp.country = np.array(countries, dtype="U128") if countries else np.array([], dtype="U1")
    bp.resolution = np.array(res_values, dtype="U32") if res_values else np.array([], dtype="U1")
    bp.nsta = len(ordered)
    return bp


def _build_npz_bundle(results: List[Tuple[Job, pd.DataFrame]], metadata: Dict[int, Dict[str, object]]) -> Optional[zdata]:
    records: Dict[str, List[object]] = {
        "time": [],
        "station_id": [],
        "value": [],
        "resolution": [],
    }

    for job, df in results:
        if df.empty:
            continue
        times = _to_datenum(df["time"])
        records["time"].extend(times.tolist())
        records["station_id"].extend([job.station_id] * len(df))
        records["value"].extend(df["value"].astype(float).tolist())
        records["resolution"].extend([job.resolution] * len(df))

    if not records["time"]:
        return None

    times = np.array(records["time"], dtype=float)
    order = np.argsort(times)
    station_ids = np.array(records["station_id"], dtype=int)[order]
    values = np.array(records["value"], dtype=float)[order]
    resolutions = np.array(records["resolution"], dtype="U32")[order]
    times_sorted = times[order]

    bundle = zdata()
    bundle.time = times_sorted
    bundle.elev = values
    bundle.station = station_ids
    bundle.bp = _build_bp(metadata, station_ids, resolutions)
    return bundle


def run_job(job: Job, *, out_dir: Path, cfg: dict) -> Optional[Tuple[pd.DataFrame, Optional[Path]]]:
    tries = int(cfg.get("request_retries", 3))
    sleep = float(cfg.get("request_sleep", 1.5))
    timeout = int(cfg.get("request_timeout", 60))
    station_id = job.station_id
    res = job.resolution
    mode = job.mode
    write_csv = bool(cfg.get("write_csv", True))

    def _write_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Path]]:
        # Apply window in client for WOCE (ERDDAP applies in server already)
        start = _safe_to_datetime(cfg.get("start"))
        end = _safe_to_datetime(cfg.get("end"))
        df2 = window_filter(df, start, end)
        df2["station_id"] = int(station_id)
        df2["resolution"] = res
        df2["station_name"] = job.station_name
        df2["country"] = job.country
        df2["lon"] = job.lon if job.lon is not None else np.nan
        df2["lat"] = job.lat if job.lat is not None else np.nan
        out_path: Optional[Path] = None
        if write_csv:
            out_dir.mkdir(parents=True, exist_ok=True)
            name_comp = _sanitize_component(job.station_name)
            country_comp = _sanitize_component(job.country)
            out_path = out_dir / f"{int(station_id)}_{name_comp}_{country_comp}.csv"
            df2.to_csv(out_path, index=False)
        return df2, out_path

    try:
        if mode == "woce":
            text = _fetch_text(job.url, tries=tries, sleep=sleep, timeout=timeout)
            if res == "daily_fast":
                df = parse_woce_daily(text)
            else:
                df = parse_woce_hourly(text)
            return _write_df(df)
        else:  # erddap
            df = _fetch_csv(job.url, tries=tries, sleep=sleep, timeout=timeout)
            # Normalize to [time, value]
            tcol = None
            for c in df.columns:
                if str(c).strip().lower() == "time":
                    tcol = c
                    break
            if tcol is None:
                raise ValueError("No 'time' column in ERDDAP response")
            cand_list = cfg.get("value_columns", [])
            valcol = _select_value_column(df, tcol, cand_list)
            df[valcol] = pd.to_numeric(df[valcol], errors="coerce")
            df = df[[tcol, valcol]]
            df.columns = ["time", "value"]
            df["time"] = _normalize_time(df["time"])
            df = df.dropna(subset=["time"]).reset_index(drop=True)
            return _write_df(df)
    except Exception as e:
        # If WOCE failed, optionally try ERDDAP as a fallback
        if mode == "woce" and bool(cfg.get("erddap_fallback", True)):
            try:
                base = str(cfg.get("erddap_base"))
                ds_map: Dict[str, str] = cfg.get("erddap_datasets", {})
                dataset = ds_map.get(res, None)
                if dataset is None:
                    raise RuntimeError("No ERDDAP dataset configured for this resolution")
                start = _safe_to_datetime(cfg.get("start"))
                end = _safe_to_datetime(cfg.get("end"))
                varlist = str(cfg.get("erddap_vars", "time,sea_level,uhslc_id"))
                url = build_erddap_url(base, dataset, varlist, station_id, start, end)
                df = _fetch_csv(url, tries=tries, sleep=sleep, timeout=timeout)
                # Normalize
                tcol = None
                for c in df.columns:
                    if str(c).strip().lower() == "time":
                        tcol = c
                        break
                if tcol is None:
                    raise ValueError("No 'time' column in ERDDAP response")
                cand_list = cfg.get("value_columns", [])
                valcol = _select_value_column(df, tcol, cand_list)
                df[valcol] = pd.to_numeric(df[valcol], errors="coerce")
                df = df[[tcol, valcol]]
                df.columns = ["time", "value"]
                df["time"] = _normalize_time(df["time"])
                df = df.dropna(subset=["time"]).reset_index(drop=True)
                return _write_df(df)
            except Exception as e2:
                sys.stderr.write(f"[WARN] station {station_id}: {e}\n")
                sys.stderr.write(f"[WARN] station {station_id}: ERDDAP fallback failed: {e2}\n")
                return None
        else:
            sys.stderr.write(f"[WARN] station {station_id}: {e}\n")
            return None


def build_jobs(df_cat: pd.DataFrame, cfg: dict) -> Tuple[List[Job], Dict[int, Dict[str, object]]]:
    resolution = str(cfg.get("resolution", "hourly_fast")).lower()
    mode = str(cfg.get("mode", "woce")).lower()
    jobs: List[Job] = []
    metadata: Dict[int, Dict[str, object]] = {}

    # Optional station filter
    station_ids = cfg.get("station_ids")
    if station_ids is not None:
        wanted = set(int(x) for x in station_ids)
        df_cat = df_cat[df_cat["station_id"].astype(int).isin(wanted)]

    # Optional max
    max_n = cfg.get("max_stations")
    if max_n is not None:
        df_cat = df_cat.head(int(max_n))

    if mode == "woce":
        col = "fast_hourly_url" if resolution == "hourly_fast" else "fast_daily_url"
        if col not in df_cat.columns:
            raise ValueError(f"Catalog missing column '{col}' required for WOCE mode")
        for r in df_cat.itertuples(index=False):
            url = getattr(r, col)
            if not isinstance(url, str) or not url:
                continue
            sid = int(r.station_id)
            name = str(getattr(r, "station_name", "")).strip()
            country = str(getattr(r, "country", "")).strip()
            lat = float(r.lat) if hasattr(r, "lat") and pd.notna(r.lat) else np.nan
            lon = float(r.lon) if hasattr(r, "lon") and pd.notna(r.lon) else np.nan
            metadata[sid] = {
                "station_id": sid,
                "station_name": name or str(sid),
                "country": country,
                "lat": lat,
                "lon": lon,
                "resolution": resolution,
            }
            jobs.append(Job(
                station_id=sid,
                url=url,
                resolution=resolution,
                mode=mode,
                station_name=name or str(sid),
                country=country,
                lat=lat if not np.isnan(lat) else None,
                lon=lon if not np.isnan(lon) else None,
            ))
    else:
        base = str(cfg.get("erddap_base"))
        ds_map: Dict[str, str] = cfg.get("erddap_datasets", {})
        dataset = ds_map.get(resolution)
        if not dataset:
            raise ValueError(f"No ERDDAP dataset configured for resolution '{resolution}'")
        varlist = str(cfg.get("erddap_vars", "time,sea_level,uhslc_id"))
        start = _safe_to_datetime(cfg.get("start"))
        end = _safe_to_datetime(cfg.get("end"))
        for r in df_cat.itertuples(index=False):
            sid = int(r.station_id)
            url = build_erddap_url(base, dataset, varlist, sid, start, end)
            name = str(getattr(r, "station_name", "")).strip()
            country = str(getattr(r, "country", "")).strip()
            lat = float(r.lat) if hasattr(r, "lat") and pd.notna(r.lat) else np.nan
            lon = float(r.lon) if hasattr(r, "lon") and pd.notna(r.lon) else np.nan
            metadata[sid] = {
                "station_id": sid,
                "station_name": name or str(sid),
                "country": country,
                "lat": lat,
                "lon": lon,
                "resolution": resolution,
            }
            jobs.append(Job(
                station_id=sid,
                url=url,
                resolution=resolution,
                mode=mode,
                station_name=name or str(sid),
                country=country,
                lat=lat if not np.isnan(lat) else None,
                lon=lon if not np.isnan(lon) else None,
            ))

    return jobs, metadata


def parse_args():
    ap = argparse.ArgumentParser(description="Download UHSLC water level time series for stations in a catalog.")
    ap.add_argument("--catalog", type=Path, help="Catalog CSV produced by build_uhslc_catalog2.py")
    ap.add_argument("--out-dir", type=Path, help="Output directory for per-station CSVs")
    ap.add_argument("--resolution", type=str, choices=["hourly_fast", "daily_fast", "hourly_rqds"],
                    help="Temporal resolution to download")
    ap.add_argument("--mode", type=str, choices=["woce", "erddap"], help="Download mode")
    ap.add_argument("--start", type=str, help="Start time (YYYY-MM-DD or ISO)")
    ap.add_argument("--end", type=str, help="End time (YYYY-MM-DD or ISO)")
    ap.add_argument("--station-ids", type=int, nargs="*", help="Only these station IDs")
    ap.add_argument("--max-stations", type=int, help="Limit number of stations (debug)")
    ap.add_argument("--timeout", type=int, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, help="HTTP retries")
    ap.add_argument("--sleep", type=float, help="Sleep seconds between retries")
    ap.add_argument("--max-workers", type=int, help="Max concurrent downloads")
    ap.add_argument("--erddap-base", type=str, help="ERDDAP base URL (for erddap mode)")
    ap.add_argument("--erddap-vars", type=str, help="CSV of variables for ERDDAP query")
    ap.add_argument("--npz-name", type=str, help="Base name for NPZ output")
    ap.add_argument("--config", type=str, help="Path to JSON config")
    ap.add_argument("--dump-config-template", action="store_true", help="Print JSON config template and exit")
    ap.add_argument("--write-combined", action="store_true", help="Also write combined CSV of all stations")
    ap.add_argument("--no-csv", action="store_true", help="Skip writing per-station CSV files")
    ap.add_argument("--no-npz", action="store_true", help="Skip writing NPZ bundle")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.dump_config_template:
        _dump_template()
        sys.exit(0)

    file_cfg = _load_json_config(getattr(args, "config", None))
    cfg = {**DEFAULT_CONFIG, **file_cfg}

    # CLI precedence
    if args.catalog is not None:
        cfg["catalog"] = os.fspath(args.catalog)
    if args.out_dir is not None:
        cfg["out_dir"] = os.fspath(args.out_dir)
    if args.resolution is not None:
        cfg["resolution"] = args.resolution
    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.start is not None:
        cfg["start"] = args.start
    if args.end is not None:
        cfg["end"] = args.end
    if args.station_ids is not None:
        cfg["station_ids"] = list(args.station_ids)
    if args.max_stations is not None:
        cfg["max_stations"] = args.max_stations
    if args.timeout is not None:
        cfg["request_timeout"] = args.timeout
    if args.retries is not None:
        cfg["request_retries"] = args.retries
    if args.sleep is not None:
        cfg["request_sleep"] = args.sleep
    if args.max_workers is not None:
        cfg["max_workers"] = args.max_workers
    if args.erddap_base is not None:
        cfg["erddap_base"] = args.erddap_base
    if args.erddap_vars is not None:
        cfg["erddap_vars"] = args.erddap_vars
    if args.write_combined:
        cfg["write_combined"] = True
    if args.npz_name is not None:
        cfg["npz_name"] = args.npz_name
    if args.no_csv:
        cfg["write_csv"] = False
    if args.no_npz:
        cfg["write_npz"] = False

    # Validate inputs
    catalog = cfg.get("catalog")
    if not catalog:
        raise SystemExit("--catalog must be provided via CLI or config")
    out_dir = Path(cfg.get("out_dir", "uhslc_data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load catalog and build jobs
    df_cat = read_catalog(Path(catalog))
    jobs, metadata = build_jobs(df_cat, cfg)
    if not jobs:
        print("No jobs to run. Check filters and catalog columns.")
        return

    # Output subdir per resolution
    res = str(cfg.get("resolution", "hourly_fast"))
    out_subdir = out_dir / res

    downloaded: List[Tuple[Job, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=int(cfg.get("max_workers", 4))) as ex:
        futs = {ex.submit(run_job, job, out_dir=out_subdir, cfg=cfg): job for job in jobs}
        for fut in as_completed(futs):
            job = futs[fut]
            result = fut.result()
            if result is None:
                continue
            df, _path = result
            if df is not None and not df.empty:
                downloaded.append((job, df))

    if not downloaded:
        print("No station data were downloaded. Nothing to write.")
        return

    if bool(cfg.get("write_combined", False)):
        out_subdir.mkdir(parents=True, exist_ok=True)
        df_all = pd.concat([df for _, df in downloaded], ignore_index=True)
        combined_path = out_subdir / "combined.csv"
        df_all.to_csv(combined_path, index=False)
        print(f"[OK] wrote combined CSV: {combined_path.resolve()}")

    if bool(cfg.get("write_npz", True)):
        bundle = _build_npz_bundle(downloaded, metadata)
        if bundle is not None:
            npz_base = cfg.get("npz_name", "uhslc_water_levels")
            npz_path = out_dir / f"{npz_base}_{res}.npz"
            savez(os.fspath(npz_path), bundle)
            print(f"[OK] wrote NPZ bundle: {npz_path.resolve()}")

    print(f"[OK] finished. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
