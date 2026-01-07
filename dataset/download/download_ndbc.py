#!/usr/bin/env python3
'''
  Download wave observations from NDBC and store them as a single npz bundle.

  The workflow mirrors download_noaa.py: configure stations, time windows, and
  output location at the top of this script, then run it to fetch, parse, and
  consolidate NDBC records.
'''

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Dict, Iterable, List

import pandas as pd

try:
    import requests
except ModuleNotFoundError as exc:
    raise SystemExit(
        "The 'requests' package is required. Install it with 'pip install requests' "
        "or load the module in your environment."
    ) from exc

from pylib import *
from numpy import ndarray


#-----------------------------------------------------------------------------
# User configuration
#-----------------------------------------------------------------------------

# Stations can be listed explicitly or resolved from a SCHISM-style bp file
# (station_wave.in). When both are provided, the intersection is used.
station_ids = ['44065', '44025', '44091']
station_file = './station_wave.in'  # optional; set to None to skip

# Time window for the download
start_date = '2020-01-01'
end_date = '2022-12-31'

# Output configuration
output_dir = 'ndbc_data'
output_name = 'NDBC_NYNJ'

# Variables to keep from the NDBC feeds (use NDBC column names)
wave_variables = {
    'WVHT': 'wvht',  # Significant wave height (m)
    'DPD': 'dpd',    # Dominant wave period (s)
    'APD': 'apd',    # Average wave period (s)
    'MWD': 'mwd',    # Mean wave direction (deg)
}


#-----------------------------------------------------------------------------
# Constants and helper structures
#-----------------------------------------------------------------------------

BASE_URL = 'https://www.ndbc.noaa.gov/view_text_file.php'
MISSING_VALUES = [99, 999, 9999, 99.0, 999.0, 9999.0]


@dataclass(frozen=True)
class Station:
    sid: str
    name: str
    lon: float | None
    lat: float | None
    depth: float | None = None


def resolve_path(base_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


#-----------------------------------------------------------------------------
# Station metadata utilities
#-----------------------------------------------------------------------------

def parse_station_wave(path: str) -> Dict[str, Station]:
    """Parse SCHISM station file entries into NDBC station metadata."""

    if not path or not fexist(path):
        return {}

    stations: Dict[str, Station] = {}
    with open(path, 'r', encoding='ascii') as fin:
        lines = [line.strip() for line in fin if line.strip() and not line.lstrip().startswith('#')]

    if len(lines) < 3:
        return {}

    entries = lines[2:]
    for entry in entries:
        if '!' in entry:
            body, comment = entry.split('!', 1)
            name = comment.strip()
        else:
            body = entry
            name = ''

        parts = body.split()
        if len(parts) < 4:
            continue

        _, slon, slat, sdepth = parts[:4]
        match = re.search(r'(\d{5})', name)
        station_id = match.group(1) if match else ''.join(ch for ch in name if ch.isdigit())
        if not station_id:
            continue

        stations[station_id] = Station(
            sid=station_id,
            name=name or station_id,
            lon=float(slon),
            lat=float(slat),
            depth=float(sdepth),
        )

    return stations


def select_stations(candidates: Dict[str, Station], requested: Iterable[str]) -> Dict[str, Station]:
    """Filter stations to those requested; create fallbacks for missing metadata."""

    selection: Dict[str, Station] = {}
    for sid in requested:
        if sid in candidates:
            selection[sid] = candidates[sid]
        else:
            selection[sid] = Station(sid=sid, name=sid, lon=None, lat=None)
    return selection


#-----------------------------------------------------------------------------
# Download and parsing helpers
#-----------------------------------------------------------------------------

def year_range(start: datetime, end: datetime) -> range:
    return range(start.year, end.year + 1)


def build_year_urls(station_id: str, year: int) -> List[str]:
    return [
        f"{BASE_URL}?filename={station_id}h{year}.txt.gz&dir=data/historical/stdmet/",
        f"{BASE_URL}?filename={station_id}.txt&dir=data/realtime2/",
    ]


def fetch_ndbc_text(session: requests.Session, station_id: str, year: int) -> str | None:
    """Download yearly NDBC text; fallback to realtime feed if needed."""

    for url in build_year_urls(station_id, year):
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException:
            continue
        if not response.ok:
            continue
        text = response.content.decode('utf-8', errors='ignore')
        if len(text.splitlines()) > 5:
            return text
    return None


def parse_ndbc_text(text: str) -> pd.DataFrame:
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return pd.DataFrame()

    header = lines[0].strip().split()
    data_lines = '\n'.join(lines[2:])
    df = pd.read_csv(
        StringIO(data_lines),
        delim_whitespace=True,
        names=header,
        na_values=MISSING_VALUES,
    )
    return df


def normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {'YY': 'year', '#YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'HH': 'hour', 'mm': 'minute'}
    for raw, alias in mapping.items():
        if raw in df.columns:
            df[alias] = df[raw]

    if not {'year', 'month', 'day', 'hour'}.issubset(df.columns):
        raise ValueError('missing time columns in NDBC file')

    if 'minute' not in df.columns:
        df['minute'] = 0

    df['datetime'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour', 'minute']],
        errors='coerce',
    ).dt.tz_localize(None)
    df = df.dropna(subset=['datetime'])
    return df


def extract_wave_variables(df: pd.DataFrame) -> pd.DataFrame:
    required = list(wave_variables.keys())
    missing = [var for var in required if var not in df.columns]
    if missing:
        raise ValueError(f'missing columns: {missing}')

    columns = ['datetime'] + required
    out = df[columns].copy()
    out = out.dropna(subset=required, how='all')
    out = out.drop_duplicates(subset='datetime')
    return out


#-----------------------------------------------------------------------------
# Consolidation utilities
#-----------------------------------------------------------------------------

def to_datenum(series: pd.Series) -> ndarray:
    return array(date2num(series.dt.to_pydatetime()))


def build_station_metadata(stations: Dict[str, Station]) -> zdata:
    meta = zdata()
    ordered = [stations[sid] for sid in stations]
    meta.station = array([s.sid for s in ordered])
    meta.name = array([s.name for s in ordered])
    meta.lon = array([s.lon if s.lon is not None else nan for s in ordered])
    meta.lat = array([s.lat if s.lat is not None else nan for s in ordered])
    meta.depth = array([s.depth if s.depth is not None else nan for s in ordered])
    meta.nsta = len(ordered)
    return meta


def append_records(bundle: Dict[str, List[float | str]], frame: pd.DataFrame, station_id: str) -> None:
    times = to_datenum(frame['datetime'])
    bundle['time'].extend(times)
    bundle['station'].extend([station_id] * len(frame))
    for src, alias in wave_variables.items():
        bundle[alias].extend(frame[src].astype(float).tolist())


def aggregate_records(records: Dict[str, List[float | str]]) -> Dict[str, ndarray]:
    times = array(records['time'], dtype=float)
    order = argsort(times)

    data: Dict[str, ndarray] = {}
    for key, values in records.items():
        if key == 'time':
            arr = times
        else:
            arr = array(values)
            if key == 'station':
                arr = arr.astype('U16')
        data[key] = arr[order]
    return data


#-----------------------------------------------------------------------------
# Main driver
#-----------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    add_var(['station_ids', 'station_file', 'output_dir', 'output_name'], [station_ids, station_file, output_dir, output_name], globals())

    station_path = resolve_path(script_dir, globals()['station_file']) if globals()['station_file'] else None
    stations = parse_station_wave(station_path) if station_path else {}
    selected = select_stations(stations, globals()['station_ids'])

    out_dir = resolve_path(script_dir, globals()['output_dir'])
    os.makedirs(out_dir, exist_ok=True)

    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)

    records: Dict[str, List[float | str]] = {key: [] for key in ['time', 'station', *wave_variables.values()]}

    session = requests.Session()
    for sid in selected:
        for year in year_range(start_dt, end_dt):
            text = fetch_ndbc_text(session, sid, year)
            if text is None:
                print(f'skip station {sid}, year {year}: download failed')
                continue

            try:
                df = parse_ndbc_text(text)
                if df.empty:
                    print(f'skip station {sid}, year {year}: empty file')
                    continue
                df = normalize_time_columns(df)
                df = extract_wave_variables(df)
            except Exception as exc:
                print(f'skip station {sid}, year {year}: {exc}')
                continue

            mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
            df = df.loc[mask]
            if df.empty:
                continue

            append_records(records, df, sid)

    if len(records['time']) == 0:
        raise SystemExit('No NDBC observations were gathered for the requested window/stations.')

    data = aggregate_records(records)

    S = zdata()
    for key, values in data.items():
        S.attr(key, values)
    S.bp = build_station_metadata(selected)

    save_path = os.path.join(out_dir, f"{globals()['output_name']}.npz")
    savez(save_path, S)
    print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
