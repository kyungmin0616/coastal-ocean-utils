#!/usr/bin/env python3
# build_uhslc_catalog.py
#
# Build a UHSLC station catalog (worldwide or regional) from ERDDAP.
# Cleans ERDDAP footer rows (e.g., last_rq_date == 'UTC'), coerces types,
# and outputs a tidy CSV aligned to your schema.
#
# Usage examples:
#   python build_uhslc_catalog.py --out uhslc_catalog.csv
#   python build_uhslc_catalog.py --out uhslc_eastasia.csv --eastasia
#   python build_uhslc_catalog.py --bbox 10 50 100 160 --include-hourly-fast --include-rqds
#
# Requirements: pandas, requests

import io
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import pandas as pd
import requests


ERDDAP_BASE = "https://uhslc.soest.hawaii.edu/erddap/tabledap"

# Named bbox presets for convenience; extend/edit as needed
PRESET_BBOXES = {
    "eastasia": (10.0, 50.0, 100.0, 160.0),
}

# Editable in-file defaults. These are used unless overridden by a JSON config
# via --config and/or CLI flags. CLI has the highest precedence.
DEFAULT_CONFIG = {
    "erddap_base": ERDDAP_BASE,
    "out": "uhslc_eastasia_catalog.csv",
    "preset": "eastasia",          # e.g., "eastasia"
    "bbox": None,            # [min_lat, max_lat, min_lon, max_lon]
    "include_hourly_fast": True,
    "include_rqds": True,
    "request_timeout": 60,   # seconds
    "request_retries": 3,
    "request_sleep": 1.5,    # seconds
    "max_workers": 2,
}

# Common station-metadata fields exposed as subsetVariables on ERDDAP info pages
BASE_FIELDS = [
    "station_name",
    "station_country",
    "station_country_code",
    "record_id",
    "uhslc_id",
    "gloss_id",
    "ssc_id",
    "latitude",
    "longitude",
]

FAST_DAILY_DATASET   = "global_daily_fast"   # includes last_rq_date
FAST_HOURLY_DATASET  = "global_hourly_fast"  # optional union
RQDS_HOURLY_DATASET  = "global_hourly_rqds"  # optional union


def _fetch_csv(url: str, tries: int = 3, sleep: float = 1.5, timeout: int = 60) -> pd.DataFrame:
    last_err = None
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=timeout, headers={"Accept": "text/csv"})
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"Failed to GET: {url}\n{last_err}")


def _build_query_url(
    dataset: str,
    fields: List[str],
    bbox: Optional[Tuple[float, float, float, float]] = None,
    distinct: bool = True,
) -> str:
    varlist = ",".join(fields)
    url = f"{ERDDAP_BASE}/{dataset}.csv?{varlist}"
    if distinct:
        url += "&distinct()"
    if bbox:
        min_lat, max_lat, min_lon, max_lon = bbox
        url += f"&latitude>={min_lat}&latitude<={max_lat}&longitude>={min_lon}&longitude<={max_lon}"
    return url


def _normalize_columns(df: pd.DataFrame, keep_last_rq_date: bool) -> pd.DataFrame:
    # Rename to internal schema
    rename_map = {
        "station_name": "station_name",
        "station_country": "country",
        "latitude": "lat",
        "longitude": "lon",
        "uhslc_id": "station_id",
        "record_id": "record_id",
        "gloss_id": "gloss_id",
        "ssc_id": "ssc_id",
        "station_country_code": "station_country_code",
    }
    df = df.rename(columns=rename_map)

    # Ensure expected columns exist
    for c in ["station_name","country","lat","lon","station_id","record_id","gloss_id","ssc_id","station_country_code"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Ensure last_rq_date presence (may be filled from FAST daily)
    if "last_rq_date" not in df.columns:
        df["last_rq_date"] = pd.NA

    return df


def _decorate_urls(df: pd.DataFrame) -> pd.DataFrame:
    def station_url(stn):
        try:
            return f"https://uhslc.soest.hawaii.edu/stations/?stn={int(stn)}"
        except Exception:
            return pd.NA

    def fd_hourly_url(stn):
        try:
            return f"https://uhslc.soest.hawaii.edu/woce/h{int(stn)}.dat"
        except Exception:
            return pd.NA

    def fd_daily_url(stn):
        try:
            return f"https://uhslc.soest.hawaii.edu/woce/d{int(stn)}.dat"
        except Exception:
            return pd.NA

    df["source_network"] = "UHSLC"
    # Leave as .apply for clarity; functions are cheap and NA-safe
    df["authority_url"] = df["station_id"].apply(station_url)
    df["fast_daily_url"] = df["station_id"].apply(fd_daily_url)
    df["fast_hourly_url"] = df["station_id"].apply(fd_hourly_url)
    df["temporal_resolution"] = "Hourly (Fast), Daily (Fast), Monthly/Hourly (RQ)"
    return df


def _clean_and_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop footer/empty rows, trim strings."""
    # Trim obvious string columns (preserve NA using pandas StringDtype)
    for c in ["station_name", "country", "authority_url", "fast_daily_url", "fast_hourly_url", "temporal_resolution", "ssc_id"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # Coerce numerics
    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce")

    # DROP known ERDDAP footer rows (e.g., trailing timezone line like last_rq_date == 'UTC')
    if "last_rq_date" in df.columns:
        mask_footer = (df["last_rq_date"].astype(str).str.upper() == "UTC") & (
            df[["station_id","station_name","country","lat","lon"]].isna().all(axis=1)
        )
        df = df[~mask_footer]

    # DROP rows with no name and no id and no coordinates
    empty_core = df[["station_id","station_name","lat","lon"]].isna().all(axis=1)
    df = df[~empty_core]

    # Optional sanity: require coordinates
    df = df.dropna(subset=["lat","lon"])

    # Dedupe after cleaning
    if "station_id" in df.columns:
        df = df.drop_duplicates(subset=["station_id"], keep="first")

    # Sort by station_id if available
    if "station_id" in df.columns:
        df = df.sort_values("station_id").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def build_catalog(
    out_path: str,
    bbox: Optional[Tuple[float, float, float, float]],
    include_hourly_fast: bool,
    include_rqds: bool,
    *,
    request_timeout: int = 60,
    request_retries: int = 3,
    request_sleep: float = 1.5,
    max_workers: int = 2,
) -> pd.DataFrame:
    # 1) FAST DAILY (primary; has last_rq_date)
    fields_fast = BASE_FIELDS + ["last_rq_date"]
    url_fast = _build_query_url(FAST_DAILY_DATASET, fields_fast, bbox=bbox, distinct=True)
    fast = _fetch_csv(url_fast, tries=request_retries, sleep=request_sleep, timeout=request_timeout)
    fast = _normalize_columns(fast, keep_last_rq_date=True)

    dfs = [fast]

    # 2) Optionally fetch other datasets concurrently
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        if include_hourly_fast:
            url_fh = _build_query_url(FAST_HOURLY_DATASET, BASE_FIELDS, bbox=bbox, distinct=True)
            tasks.append(("fh", ex.submit(_fetch_csv, url_fh, request_retries, request_sleep, request_timeout)))
        if include_rqds:
            url_rq = _build_query_url(RQDS_HOURLY_DATASET, BASE_FIELDS, bbox=bbox, distinct=True)
            tasks.append(("rq", ex.submit(_fetch_csv, url_rq, request_retries, request_sleep, request_timeout)))

        for tag, fut in tasks:
            df = fut.result()
            df = _normalize_columns(df, keep_last_rq_date=False)
            dfs.append(df)

    # 4) Union (FAST DAILY first so its last_rq_date is kept), then clean/types
    merged = pd.concat(dfs, ignore_index=True)
    merged = _decorate_urls(merged)
    merged = _clean_and_types(merged)

    # 5) Reorder columns and write
    cols = [
        "station_id","station_name","country","station_country_code",
        "lat","lon",
        "source_network","authority_url","temporal_resolution",
        "fast_daily_url","fast_hourly_url",
        "record_id","gloss_id","ssc_id","last_rq_date"
    ]
    for c in cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    merged = merged[cols].reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    return merged


def parse_args():
    ap = argparse.ArgumentParser(description="Build a UHSLC station catalog from ERDDAP.")
    ap.add_argument("--out", type=str, help="Output CSV path")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LAT","MAX_LAT","MIN_LON","MAX_LON"),
                    help="Restrict by bounding box (e.g., 10 50 100 160)")
    ap.add_argument("--eastasia", action="store_true",
                    help="Shortcut for East Asia bbox (10–50N, 100–160E)")
    ap.add_argument("--preset", type=str, help=f"Named bbox preset: one of {sorted(PRESET_BBOXES.keys())}")
    ap.add_argument("--include-hourly-fast", action="store_true",
                    help="Union distinct stations from global_hourly_fast")
    ap.add_argument("--include-rqds", action="store_true",
                    help="Union distinct stations from global_hourly_rqds (Research-Quality)")
    ap.add_argument("--config", type=str, help="Path to JSON config with defaults")
    ap.add_argument("--erddap-base", type=str, help="Override ERDDAP base URL")
    ap.add_argument("--timeout", type=int, help="HTTP timeout seconds (default from config)")
    ap.add_argument("--retries", type=int, help="HTTP retries (default from config)")
    ap.add_argument("--sleep", type=float, help="Sleep seconds between retries (default from config)")
    ap.add_argument("--max-workers", type=int, help="Max threads for optional fetches (default from config)")
    ap.add_argument("--dump-config-template", action="store_true", help="Print JSON config template and exit")
    return ap.parse_args()


def _dump_template():
    import json
    print(json.dumps(DEFAULT_CONFIG, indent=2))


def _load_json_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    import json, os
    p = os.fspath(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    if getattr(args, "dump_config_template", False):
        _dump_template()
        sys.exit(0)

    file_cfg = _load_json_config(getattr(args, "config", None))
    cfg = {**DEFAULT_CONFIG, **file_cfg}

    # Resolve ERDDAP base
    base_override = getattr(args, "erddap_base", None)
    if base_override:
        cfg["erddap_base"] = base_override
    # Update global used by URL builder
    global ERDDAP_BASE
    ERDDAP_BASE = cfg.get("erddap_base", ERDDAP_BASE)

    # Resolve bbox via precedence: CLI eastasia/preset > CLI bbox > cfg preset > cfg bbox
    bbox = None
    if getattr(args, "eastasia", False):
        bbox = PRESET_BBOXES.get("eastasia")
    elif getattr(args, "preset", None):
        bbox = PRESET_BBOXES.get(args.preset.lower())
    elif getattr(args, "bbox", None):
        bbox = tuple(args.bbox)  # type: ignore
    elif cfg.get("preset"):
        bbox = PRESET_BBOXES.get(str(cfg["preset"]).lower())
    elif cfg.get("bbox"):
        bb = cfg["bbox"]
        bbox = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))

    # Resolve booleans
    include_hourly_fast = args.include_hourly_fast or bool(cfg.get("include_hourly_fast", False))
    include_rqds = args.include_rqds or bool(cfg.get("include_rqds", False))

    # Resolve output
    out_path = args.out if args.out else cfg.get("out", DEFAULT_CONFIG["out"])  # type: ignore

    # Networking/concurrency
    request_timeout = args.timeout if args.timeout is not None else int(cfg.get("request_timeout", 60))
    request_retries = args.retries if args.retries is not None else int(cfg.get("request_retries", 3))
    request_sleep = args.sleep if args.sleep is not None else float(cfg.get("request_sleep", 1.5))
    max_workers = args.max_workers if args.max_workers is not None else int(cfg.get("max_workers", 2))

    df = build_catalog(
        out_path=out_path,
        bbox=bbox,
        include_hourly_fast=include_hourly_fast,
        include_rqds=include_rqds,
        request_timeout=request_timeout,
        request_retries=request_retries,
        request_sleep=request_sleep,
        max_workers=max_workers,
    )
    print(f"[OK] wrote {len(df)} rows → {out_path}")
    if bbox:
        print(f"[info] bbox used: {bbox}")
    if args.include_hourly_fast:
        print("[info] included: global_hourly_fast")
    if args.include_rqds:
        print("[info] included: global_hourly_rqds")


if __name__ == "__main__":
    main()
