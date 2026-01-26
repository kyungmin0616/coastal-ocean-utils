#!/usr/bin/env python3
"""
analyze_bc_stations.py

Analyze MLIT station discharge CSVs (station_*.csv) and compute:
- time coverage (start/end)
- availability + nan ratio within coverage
- max missing gap (hours) within coverage
Then, per river_group, compute common overlap window across stations and
recommend a "best" station for boundary condition (BC).

Expected station CSV columns (from your downloader outputs):
- station_id
- datetime
- value_cms
Optional metadata columns:
- river_group, river_system_jp, river_name_jp, station_name_jp, lat_dd, lon_dd

Usage examples:
  python analyze_bc_stations.py --in-dir ./data --out metrics.csv
  python analyze_bc_stations.py --in-dir ./data --sentinel -10000 --dt-col datetime
Notes:
  - Expects hourly time series; missing values are filtered via sentinel.
  - The output CSV includes per-station metrics and recommended BC station per group.
"""

from __future__ import annotations

import argparse
import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------

def month_label(dt: pd.Timestamp) -> str:
    # e.g., 2021-Jan
    return dt.strftime("%Y-%b")


def safe_to_datetime(s: pd.Series) -> pd.Series:
    # fast + robust
    return pd.to_datetime(s, errors="coerce", utc=False)


def longest_nan_run_hours(is_nan: np.ndarray) -> int:
    """
    Compute longest consecutive True-run length in boolean array.
    """
    if is_nan.size == 0:
        return 0
    max_run = 0
    run = 0
    for v in is_nan:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return int(max_run)


def infer_discharge_col(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    # Your downloaded format uses value_cms
    if "value_cms" in df.columns:
        return "value_cms"

    # fallback guesses
    for c in df.columns:
        cl = c.lower()
        if "discharge" in cl or "flow" in cl:
            return c
        if "cms" in cl or "m3/s" in cl or "m3s" in cl:
            return c
        if cl in ("q", "q_cms", "q_m3s"):
            return c

    raise ValueError(f"Could not infer discharge column. Columns={list(df.columns)}")


def station_files(in_dir: str) -> List[str]:
    """
    Only ingest true station time-series outputs.
    This avoids crashing on metric CSVs you saved into the same folder.
    """
    pat = os.path.join(in_dir, "station_*.csv")
    files = sorted(glob.glob(pat))
    return files


@dataclass
class StationMetrics:
    river_group: str
    station_id: str

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    coverage_label: str

    available_hours: int
    expected_hours: int
    nan_ratio: float
    max_gap_hours: int

    # metadata (optional)
    river_system_jp: Optional[str] = None
    river_name_jp: Optional[str] = None
    station_name_jp: Optional[str] = None
    lat_dd: Optional[float] = None
    lon_dd: Optional[float] = None


def load_station_series(
    path: str,
    dt_col: str,
    q_col: Optional[str],
    sentinel: float,
) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path, dtype={"station_id": str})
    if dt_col not in df.columns:
        raise ValueError(
            f"{os.path.basename(path)}: cannot find datetime column '{dt_col}'. "
            f"Columns={list(df.columns)}"
        )

    if q_col is None:
        q_col = infer_discharge_col(df)
    if q_col not in df.columns:
        raise ValueError(
            f"{os.path.basename(path)}: cannot find discharge column '{q_col}'. "
            f"Columns={list(df.columns)}"
        )

    df[dt_col] = safe_to_datetime(df[dt_col])
    df = df.dropna(subset=[dt_col]).copy()

    # mask sentinel values (e.g., -10000)
    q = pd.to_numeric(df[q_col], errors="coerce")
    q = q.mask(q <= sentinel)  # sentinel and any more-negative junk
    df[q_col] = q

    # dedupe times (keep last)
    df = df.sort_values(dt_col)
    df = df.drop_duplicates(subset=[dt_col], keep="last")

    return df, dt_col, q_col


def compute_metrics_for_station(
    df: pd.DataFrame,
    dt_col: str,
    q_col: str,
    station_id: str,
    river_group: str,
) -> StationMetrics:
    start_time = df[dt_col].min()
    end_time = df[dt_col].max()

    # Build expected hourly index across the station's OWN coverage
    # Note: inclusive end (typical for hourly series)
    full_index = pd.date_range(start=start_time, end=end_time, freq="h")

    s = df.set_index(dt_col)[q_col].reindex(full_index)
    is_nan = s.isna().to_numpy()

    expected_hours = int(len(full_index))
    available_hours = int(s.notna().sum())
    nan_ratio = float(np.mean(is_nan)) if expected_hours > 0 else float("nan")
    max_gap_hours = longest_nan_run_hours(is_nan)

    coverage_label = f"{month_label(start_time)} to {month_label(end_time)}"

    # metadata if present
    meta = {}
    for k in ("river_system_jp", "river_name_jp", "station_name_jp", "lat_dd", "lon_dd"):
        if k in df.columns:
            v = df[k].dropna()
            meta[k] = v.iloc[0] if len(v) else None
        else:
            meta[k] = None

    return StationMetrics(
        river_group=river_group,
        station_id=station_id,
        start_time=start_time,
        end_time=end_time,
        coverage_label=coverage_label,
        available_hours=available_hours,
        expected_hours=expected_hours,
        nan_ratio=nan_ratio,
        max_gap_hours=max_gap_hours,
        river_system_jp=meta["river_system_jp"],
        river_name_jp=meta["river_name_jp"],
        station_name_jp=meta["station_name_jp"],
        lat_dd=meta["lat_dd"],
        lon_dd=meta["lon_dd"],
    )


def compute_overlap_window(group_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    """
    overlap_start = latest start among stations
    overlap_end   = earliest end among stations
    """
    overlap_start = pd.to_datetime(group_df["start_time"]).max()
    overlap_end = pd.to_datetime(group_df["end_time"]).min()

    if pd.isna(overlap_start) or pd.isna(overlap_end) or overlap_end <= overlap_start:
        return overlap_start, overlap_end, 0

    overlap_hours = int(len(pd.date_range(overlap_start, overlap_end, freq="h")))
    return overlap_start, overlap_end, overlap_hours


def metrics_within_overlap(
    path: str,
    dt_col: str,
    q_col: Optional[str],
    sentinel: float,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> Tuple[float, int]:
    """
    Recompute nan_ratio and max_gap within the overlap window only.
    """
    df, dt_col, q_col = load_station_series(path, dt_col=dt_col, q_col=q_col, sentinel=sentinel)
    df = df[(df[dt_col] >= overlap_start) & (df[dt_col] <= overlap_end)].copy()

    if df.empty:
        # fully missing in overlap
        overlap_index = pd.date_range(overlap_start, overlap_end, freq="h")
        return 1.0, int(len(overlap_index))

    overlap_index = pd.date_range(overlap_start, overlap_end, freq="h")
    s = df.set_index(dt_col)[q_col].reindex(overlap_index)

    is_nan = s.isna().to_numpy()
    nan_ratio = float(np.mean(is_nan)) if len(overlap_index) else float("nan")
    max_gap = longest_nan_run_hours(is_nan)
    return nan_ratio, max_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing station_*.csv")
    ap.add_argument("--master_csv", default=None, help="Optional: stations_master.csv (not required)")
    ap.add_argument("--dt_col", default="datetime", help="Datetime column name (default: datetime)")
    ap.add_argument("--q_col", default=None, help="Discharge column name (default: auto)")
    ap.add_argument("--sentinel", type=float, default=-9999.0, help="Mask values <= sentinel (default: -9999)")
    ap.add_argument(
        "--max_gap_interp_safe",
        type=int,
        default=72,
        help="If max_gap_hours within overlap <= this, mark interp_safe True (default: 72 hours)",
    )
    ap.add_argument("--out_csv", default=None, help="Write metrics table to CSV")
    args = ap.parse_args()

    files = station_files(args.in_dir)
    if not files:
        raise SystemExit(f"No station_*.csv found under: {args.in_dir}")

    metrics: List[StationMetrics] = []

    for path in files:
        base = os.path.basename(path)
        # station_3020....csv
        station_id = base.replace("station_", "").replace(".csv", "")

        df, dt_col, q_col = load_station_series(
            path,
            dt_col=args.dt_col,
            q_col=args.q_col,
            sentinel=args.sentinel,
        )

        # river_group: from file if present, else "Unknown"
        if "river_group" in df.columns:
            rg = df["river_group"].dropna()
            river_group = rg.iloc[0] if len(rg) else "Unknown"
        else:
            river_group = "Unknown"

        m = compute_metrics_for_station(
            df=df,
            dt_col=dt_col,
            q_col=q_col,
            station_id=station_id,
            river_group=river_group,
        )
        metrics.append(m)

    # Metrics table
    met_df = pd.DataFrame([m.__dict__ for m in metrics])

    # Compact printing columns (but keep full timestamps in CSV)
    show = met_df.copy()
    #show["start_time"] = pd.to_datetime(show["start_time"])
    #show["end_time"] = pd.to_datetime(show["end_time"])

    show["start_time"] = pd.to_datetime(show["start_time"]).dt.strftime("%Y-%b")
    show["end_time"]   = pd.to_datetime(show["end_time"]).dt.strftime("%Y-%b")

    # Print station quality metrics
    show_cols = [
        "river_group", "station_id",
        "coverage_label",
        "start_time", "end_time",
        "available_hours", "expected_hours",
        "nan_ratio", "max_gap_hours",
        "river_system_jp", "river_name_jp", "station_name_jp",
        "lat_dd", "lon_dd",
    ]
    show_cols = [c for c in show_cols if c in show.columns]

    print("\n=== Station quality metrics (with time coverage) ===")
    # sort: best availability first
    show_sorted = show[show_cols].sort_values(
        by=["river_group", "nan_ratio", "max_gap_hours", "expected_hours"],
        ascending=[True, True, True, False],
    )
    with pd.option_context("display.max_rows", 500, "display.max_columns", 200, "display.width", 200):
        print(show_sorted.to_string(index=False))

    # Recommended BC station per river (based on overlap window)
    rec_rows = []
    for river_group, g in met_df.groupby("river_group"):
        overlap_start, overlap_end, overlap_hours = compute_overlap_window(g)
        if overlap_hours <= 0:
            rec_rows.append({
                "river_group": river_group,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_hours": overlap_hours,
                "best_station_id": None,
                "best_nan_ratio": None,
                "best_max_gap_hours": None,
                "interp_safe": False,
            })
            continue

        # Evaluate each station within overlap
        candidates = []
        for _, row in g.iterrows():
            sid = str(row["station_id"])
            path = os.path.join(args.in_dir, f"station_{sid}.csv")
            if not os.path.exists(path):
                continue

            nan_ratio_o, max_gap_o = metrics_within_overlap(
                path=path,
                dt_col=args.dt_col,
                q_col=args.q_col,
                sentinel=args.sentinel,
                overlap_start=overlap_start,
                overlap_end=overlap_end,
            )
            candidates.append((sid, nan_ratio_o, max_gap_o))

        if not candidates:
            rec_rows.append({
                "river_group": river_group,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_hours": overlap_hours,
                "best_station_id": None,
                "best_nan_ratio": None,
                "best_max_gap_hours": None,
                "interp_safe": False,
            })
            continue

        # choose best: lowest nan_ratio in overlap, then lowest max gap
        candidates.sort(key=lambda x: (x[1], x[2]))
        best_sid, best_nan, best_gap = candidates[0]
        interp_safe = bool(best_gap <= args.max_gap_interp_safe)

        rec_rows.append({
            "river_group": river_group,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_hours": float(overlap_hours),
            "best_station_id": best_sid,
            "best_nan_ratio": float(best_nan),
            "best_max_gap_hours": int(best_gap),
            "interp_safe": interp_safe,
        })

    rec_df = pd.DataFrame(rec_rows).sort_values("river_group")

    # Make overlap dates compact for printing (YYYY-Mon)
    rec_show = rec_df.copy()
    rec_show["overlap_start_label"] = pd.to_datetime(rec_show["overlap_start"]).dt.strftime("%Y-%b")
    rec_show["overlap_end_label"] = pd.to_datetime(rec_show["overlap_end"]).dt.strftime("%Y-%b")

    print("\n=== Recommended BC station per river (based on common overlap window) ===")
    show_cols2 = [
        "river_group",
        "overlap_start_label", "overlap_end_label", "overlap_hours",
        "best_station_id", "best_nan_ratio", "best_max_gap_hours", "interp_safe",
    ]
    with pd.option_context("display.max_rows", 500, "display.max_columns", 200, "display.width", 200):
        print(rec_show[show_cols2].to_string(index=False))

    # Save output CSV if requested
    if args.out_csv:
        # Save full metrics + recommendations as two CSVs
        base, ext = os.path.splitext(args.out_csv)
        metrics_path = args.out_csv
        rec_path = f"{base}_recommended{ext or '.csv'}"

        met_df_out = met_df.copy()
        met_df_out["coverage_label"] = met_df_out["coverage_label"].astype(str)

        met_df_out.to_csv(metrics_path, index=False)
        rec_df.to_csv(rec_path, index=False)

        print(f"\nWrote: {metrics_path}")
        print(f"Wrote: {rec_path}")


if __name__ == "__main__":
    main()
