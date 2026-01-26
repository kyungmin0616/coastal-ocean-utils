#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MLIT river.go.jp hourly discharge downloader (KIND=6)

Workflow (matches the website):
  1) Availability page:
       https://www1.river.go.jp/cgi-bin/SrchWaterData.exe?ID=<station>&KIND=6&PAGE=0
     -> scrape available years
  2) Monthly table page:
       https://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=6&ID=<station>&BGNDATE=YYYYMMDD&ENDDATE=YYYYMMDD&KAWABOU=NO
     -> find disk-file link like:
       https://www1.river.go.jp/dat/dload/download/<token>.dat
  3) Download .dat and parse to tidy long-format CSV

Outputs:
  - per-station CSV in out_dir/station_<ID>.csv
  - optional combined CSV in out_dir/all_stations_long.csv

Usage:
  # All available data (auto-discover years/months):
  python download_River.py --csv stations_master.csv --out_dir out_mlit

  # Specific range:
  python download_River.py --csv stations_master.csv --start 2023-11-01 --end 2023-12-31 --out_dir out_mlit

Notes:
  - If you put a space after "\" in your shell, arguments break. Use backslash at end-of-line with no trailing spaces.
"""

from __future__ import annotations

import argparse
import calendar
import csv as pycsv
import datetime as dt
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://www1.river.go.jp"
URL_SRCH = BASE + "/cgi-bin/SrchWaterData.exe"
URL_DSP  = BASE + "/cgi-bin/DspWaterData.exe"

# KIND=6: 時刻流量 (hourly discharge)
KIND_HOURLY_DISCHARGE = 6


@dataclass
class Station:
    river_group: str
    river_system_jp: str
    river_name_jp: str
    station_name_jp: str
    station_id: str
    lat_dd: float
    lon_dd: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download MLIT hourly discharge (KIND=6) from river.go.jp")
    p.add_argument("--csv", required=True, help="Station master CSV (must include station_id, station_name_jp, river_group, ...)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD (optional). If omitted, download all available.")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (optional). If omitted, download all available.")
    p.add_argument("--sleep", type=float, default=0.6, help="Base sleep between requests (seconds)")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout (seconds)")
    p.add_argument("--max_years", type=int, default=200, help="Safety limit for number of years per station when auto mode")
    p.add_argument("--save_raw_failed", action="store_true", help="Save raw HTML/.dat when parsing fails")
    p.add_argument("--combined_csv", action="store_true", help="Also write one combined long CSV for all stations")
    return p.parse_args()


def yyyymmdd(d: dt.date) -> str:
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"


def month_range(year: int, month: int) -> Tuple[dt.date, dt.date]:
    last = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), dt.date(year, month, last)


def daterange_months(start: dt.date, end: dt.date) -> List[Tuple[dt.date, dt.date]]:
    """Return list of (month_start, month_end) chunks covering [start, end]."""
    chunks: List[Tuple[dt.date, dt.date]] = []
    cur = dt.date(start.year, start.month, 1)
    while cur <= end:
        ms, me = month_range(cur.year, cur.month)
        # clip to requested range
        cs = max(ms, start)
        ce = min(me, end)
        chunks.append((cs, ce))
        # next month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    return chunks


def build_session() -> requests.Session:
    s = requests.Session()

    retry = Retry(
        total=7,
        connect=7,
        read=7,
        backoff_factor=0.6,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # “Browser-like” headers to reduce 403.
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": BASE + "/",
    })
    return s


def warmup(s: requests.Session, timeout: float) -> None:
    # Hit the top page to get any cookies/session bits.
    r = s.get(BASE + "/", timeout=timeout)
    r.raise_for_status()


def load_stations(path: str) -> List[Station]:
    df = pd.read_csv(path, dtype={"station_id": str})
    need = ["river_group", "river_system_jp", "river_name_jp", "station_name_jp", "station_id", "lat_dd", "lon_dd"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    out: List[Station] = []
    for _, row in df.iterrows():
        out.append(Station(
            river_group=str(row["river_group"]),
            river_system_jp=str(row["river_system_jp"]),
            river_name_jp=str(row["river_name_jp"]),
            station_name_jp=str(row["station_name_jp"]),
            station_id=str(row["station_id"]),
            lat_dd=float(row["lat_dd"]),
            lon_dd=float(row["lon_dd"]),
        ))
    return out


def get_available_years(session: requests.Session, station_id: str, timeout: float) -> List[int]:
    """
    Scrape available years from SrchWaterData.exe page.
    The page shows decade rows (e.g., 201*, 202*) with blue dots.
    We parse any explicit 4-digit years that appear, and also infer years from decade markers + dots if present.

    Practical approach:
      - collect all '20xx' and '19xx' occurrences from HTML
      - de-duplicate and sort
    This is robust enough across minor HTML changes.
    """
    params = {"ID": station_id, "KIND": str(KIND_HOURLY_DISCHARGE), "PAGE": "0"}
    r = session.get(URL_SRCH, params=params, timeout=timeout)
    r.raise_for_status()
    html = r.text

    years = set(_parse_year_matrix(html))
    if not years:
        # fallback: raw year scan (can be polluted by station_id digits)
        years = set(int(y) for y in re.findall(r"(19\d{2}|20\d{2})", html))
    years = {y for y in years if 1900 <= y <= dt.date.today().year}
    return sorted(years)


def _parse_year_matrix(html: str) -> List[int]:
    """
    Parse the availability matrix (e.g., 201*, 202*) where each column is 0-9
    and cells contain /img/ari.gif or /img/ari0.gif to indicate availability.
    """
    years: List[int] = []
    lines = html.splitlines()
    for i, line in enumerate(lines):
        m = re.search(r">\s*(\d{3})\*", line)
        if not m:
            continue
        prefix = int(m.group(1))
        digit = 0
        j = i + 1
        while j < len(lines) and "</TR>" not in lines[j]:
            cell = lines[j].lower()
            if "/img/" in cell:
                if "ari.gif" in cell or "ari0.gif" in cell:
                    years.append(prefix * 10 + digit)
                digit += 1
            j += 1
    return years


def find_dat_link_from_dsp(html: str) -> Optional[str]:
    """
    Extract the /dat/dload/download/<token>.dat link from DspWaterData HTML.
    """
    m = re.search(r"(\/dat\/dload\/download\/[0-9A-Za-z]+\.dat)", html)
    if not m:
        return None
    return BASE + m.group(1)


def fetch_month_dat_link(
    session: requests.Session,
    station_id: str,
    start_d: dt.date,
    end_d: dt.date,
    timeout: float
) -> Tuple[Optional[str], str]:
    """
    Returns (dat_url or None, dsp_html).
    """
    params = {
        "KIND": str(KIND_HOURLY_DISCHARGE),
        "ID": station_id,
        "BGNDATE": yyyymmdd(start_d),
        "ENDDATE": yyyymmdd(end_d),
        "KAWABOU": "NO",
    }
    r = session.get(URL_DSP, params=params, timeout=timeout)
    r.raise_for_status()
    html = r.text
    return find_dat_link_from_dsp(html), html


def download_text(session: requests.Session, url: str, timeout: float) -> str:
    r = session.get(url, timeout=timeout, headers={"Referer": URL_DSP})
    r.raise_for_status()
    # .dat is plain text (Shift-JIS sometimes). requests usually guesses; we force a safe decode:
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def parse_hourly_dat(dat_text: str, station_id: str) -> pd.DataFrame:
    """
    Parse MLIT 'disk file' .dat for hourly discharge (KIND=6).

    Observed structure (typical):
      - header lines (Japanese) + comment lines starting with '#'
      - a header line listing "1時,2時,...,24時," (often preceded by '#')
      - data lines:
          YYYY/MM/DD, v1, f1, v2, f2, ... v24, f24
        sometimes flags are blank; sometimes trailing commas exist.

    Returns tidy long dataframe:
      datetime, value, flag, station_id
    """
    lines = [ln.strip() for ln in dat_text.splitlines() if ln.strip() != ""]

    # Find the hour header line (may start with '#')
    header_idx = None
    for i, ln in enumerate(lines):
        if ("1時" in ln) and ("24時" in ln) and ("2時" in ln) and ("23時" in ln):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate hourly header (1時..24時) in .dat")

    data_lines = lines[header_idx + 1 :]

    # parse data
    out_rows = []
    for ln in data_lines:
        # skip comment-like lines
        if ln.startswith("#"):
            continue
        if not re.match(r"^\d{4}\/\d{2}\/\d{2}\s*,", ln):
            continue

        # Use csv reader to handle extra commas/spaces
        reader = pycsv.reader(StringIO(ln))
        row = next(reader)

        # clean whitespace
        row = [x.strip() for x in row]
        if len(row) < 2:
            continue

        date_str = row[0]
        try:
            day = dt.datetime.strptime(date_str, "%Y/%m/%d").date()
        except Exception:
            continue

        # Remaining fields: either [v1,f1,v2,f2,...] or just [v1,v2,...]
        fields = row[1:]

        # If it looks like pairs (value,flag) for most hours, use pair parsing.
        # Heuristic: if there are >= 40 fields, it's almost certainly paired (24*2=48).
        paired = len(fields) >= 40

        for h in range(1, 25):
            if paired:
                vi = (h - 1) * 2
                fi = vi + 1
                v_raw = fields[vi] if vi < len(fields) else ""
                f_raw = fields[fi] if fi < len(fields) else ""
            else:
                vi = (h - 1)
                v_raw = fields[vi] if vi < len(fields) else ""
                f_raw = ""

            # empty means missing
            v = None
            if v_raw not in ("", "-", "—"):
                try:
                    v = float(v_raw)
                except Exception:
                    v = None

            t = dt.datetime(day.year, day.month, day.day, h % 24)  # 24時 -> 0:00 next day? MLIT uses 24時 as end-of-day
            # MLIT tables usually mean 24時 = 24:00 of same day; better store as next day's 00:00.
            if h == 24:
                t = dt.datetime(day.year, day.month, day.day) + dt.timedelta(days=1)

            out_rows.append({
                "station_id": station_id,
                "datetime": t,
                "value_cms": v,
                "flag": f_raw if f_raw != "" else None,
                "date": day.isoformat(),
                "hour": h,
            })

    if not out_rows:
        raise ValueError("Parsed 0 data rows from .dat (structure may differ for this station/month)")

    df = pd.DataFrame(out_rows).sort_values(["datetime"]).reset_index(drop=True)
    return df


def safe_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stations = load_stations(args.csv)

    # date handling
    auto_all = (args.start is None) or (args.end is None)
    start_date = None
    end_date = None
    if not auto_all:
        start_date = dt.datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
        if end_date < start_date:
            raise ValueError("--end must be >= --start")

    session = build_session()
    print("Warming up session...")
    warmup(session, timeout=args.timeout)

    combined_rows: List[pd.DataFrame] = []

    for idx, st in enumerate(stations, start=1):
        sid = st.station_id
        print(f"[{idx}/{len(stations)}] Fetching station {sid} ...")

        try:
            month_chunks: List[Tuple[dt.date, dt.date]] = []

            if auto_all:
                years = get_available_years(session, sid, timeout=args.timeout)
                if len(years) == 0:
                    print(f"  -> SKIP {sid}: no years detected on availability page")
                    continue
                if len(years) > args.max_years:
                    years = years[-args.max_years:]  # keep most recent if crazy large

                # Try every month in detected years.
                for y in years:
                    for m in range(1, 13):
                        ms, me = month_range(y, m)
                        month_chunks.append((ms, me))
            else:
                month_chunks = daterange_months(start_date, end_date)

            per_station_frames: List[pd.DataFrame] = []
            for (ms, me) in month_chunks:
                # Be polite / avoid rate-limits
                time.sleep(args.sleep + random.random() * 0.4)

                dat_url, dsp_html = fetch_month_dat_link(session, sid, ms, me, timeout=args.timeout)
                if not dat_url:
                    # no data for this month (or page layout changed)
                    continue

                # Download .dat
                time.sleep(args.sleep * 0.5 + random.random() * 0.2)
                dat_text = download_text(session, dat_url, timeout=args.timeout)

                # Parse
                dfm = parse_hourly_dat(dat_text, station_id=sid)

                # add metadata columns from station master
                dfm["river_group"] = st.river_group
                dfm["river_system_jp"] = st.river_system_jp
                dfm["river_name_jp"] = st.river_name_jp
                dfm["station_name_jp"] = st.station_name_jp
                dfm["lat_dd"] = st.lat_dd
                dfm["lon_dd"] = st.lon_dd
                dfm["source_dat_url"] = dat_url

                per_station_frames.append(dfm)

            if not per_station_frames:
                print(f"  -> NO DATA {sid}: no .dat links found for requested period")
                continue

            out_df = pd.concat(per_station_frames, ignore_index=True)
            out_df = out_df.sort_values("datetime").reset_index(drop=True)

            out_path = os.path.join(args.out_dir, f"station_{sid}.csv")
            out_df.to_csv(out_path, index=False)
            print(f"  -> OK {sid} ({len(out_df):,} rows)")

            if args.combined_csv:
                combined_rows.append(out_df)

        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl-C). Exiting cleanly.")
            break
        except Exception as e:
            print(f"  -> FAILED station {sid}: {e}")
            if args.save_raw_failed:
                # Save last-known artifacts if we have them
                # (We may not always have dsp_html/dat_text depending on failure point.)
                fail_dir = os.path.join(args.out_dir, "_failed_raw", sid)
                os.makedirs(fail_dir, exist_ok=True)
                # Best-effort: re-fetch one month (or availability) for debugging
                try:
                    dbg = session.get(URL_SRCH, params={"ID": sid, "KIND": str(KIND_HOURLY_DISCHARGE), "PAGE": "0"}, timeout=args.timeout)
                    safe_write(os.path.join(fail_dir, "SrchWaterData.html"), dbg.text)
                except Exception:
                    pass
            continue

    if args.combined_csv and combined_rows:
        all_df = pd.concat(combined_rows, ignore_index=True)
        all_path = os.path.join(args.out_dir, "all_stations_long.csv")
        all_df.to_csv(all_path, index=False)
        print(f"Combined written: {all_path} ({len(all_df):,} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
