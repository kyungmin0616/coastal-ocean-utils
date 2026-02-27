#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
River_plot.py

Quick plotting utility for MLIT river station CSVs (station_*.csv).
Reads station files and renders discharge time series with optional
time windows and labeling modes.

Usage examples:
  # One figure per river group (needs master CSV with river_group)
  python River_plot.py --in-dir ./data --mode group --master_csv ./stations.csv --out ./figs

  # One figure per station
  python River_plot.py --in-dir ./data --mode station --out ./figs

  # Filter to specific stations
  python River_plot.py --in-dir ./data --mode station --station-list 303051,303052 --out ./figs

  # Time window and label with station name
  python River_plot.py --in-dir ./data --mode station --start 2021-01-01 --end 2021-12-31 \
      --label station_id_name --out ./figs

Flags summary:
  --in-dir         Directory containing station_*.csv files.
  --mode           group (one plot per river_group) or station (one plot per station).
  --master_csv     Master station list CSV (required for group mode).
  --start/--end     Optional time window (YYYY-MM-DD).
  --save-dir/--out  Output directory for figures.
  --fmt            png|pdf|svg (default: png).
  --label           station_id or station_id_name.
  --station-list   Comma-separated station IDs to plot (e.g., 303051,303052).
  --mask-threshold Mask values <= this as missing (default: -9999).
  --ymin/--ymax     Optional y-axis limits.
Notes:
  - Missing values <= -9999 are masked to NaN.
  - Date parsing uses pandas; provide ISO timestamps for best results.
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -----------------------------
# Helpers
# -----------------------------
def parse_date(s: str | None) -> pd.Timestamp | None:
    if not s:
        return None
    return pd.to_datetime(s, errors="raise")


def parse_station_list(s: str | None) -> set[str] | None:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return set(items) if items else None


def load_master_csv(master_csv: str) -> pd.DataFrame:
    df = pd.read_csv(master_csv, dtype={"station_id": str})
    # Normalize expected columns if present
    if "station_id" not in df.columns:
        raise ValueError(f"master_csv must contain 'station_id'. Columns={list(df.columns)}")
    return df


def infer_datetime_col(df: pd.DataFrame) -> str:
    # Your downloader writes 'datetime'
    for c in ["datetime", "time", "timestamp", "date_time", "dt"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not infer datetime column. Columns={list(df.columns)}")


def infer_discharge_col(df: pd.DataFrame) -> str:
    """
    Your station CSVs have: value_cms, flag, date, hour, ...
    This function returns the discharge/flow column.
    """
    candidates = [
        "value_cms",     # <-- your current output
        "discharge_cms",
        "q_cms",
        "flow_cms",
        "Q",
        "q",
        "discharge",
        "flow",
        "value",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: any numeric column that looks like discharge
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # remove obvious non-Q numerics
    bad = {"lat_dd", "lon_dd", "hour"}
    numeric_cols = [c for c in numeric_cols if c not in bad]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(f"Could not infer discharge column. Columns={list(df.columns)}")


def mask_missing_q(series: pd.Series, missing_threshold: float = -9999.0) -> pd.Series:
    """
    MLIT hourly data often uses -10000 (or <= -9999) as missing.
    Mask those values to NaN.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > missing_threshold, np.nan)
    return s


def load_one_station_csv(path: str) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path, dtype={"station_id": str})
    dt_col = infer_datetime_col(df)
    q_col = infer_discharge_col(df)

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col]).sort_values(dt_col)

    df[q_col] = mask_missing_q(df[q_col])
    return df, dt_col, q_col


def choose_label(df: pd.DataFrame, label_mode: str) -> str:
    sid = str(df["station_id"].iloc[0]) if "station_id" in df.columns else "unknown"
    if label_mode == "station_id":
        return sid
    # station_id + name if available
    name = None
    for c in ["station_name_jp", "station_name", "name"]:
        if c in df.columns:
            name = df[c].iloc[0]
            break
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return sid
    return f"{sid}:{name}"


def apply_time_window(df: pd.DataFrame, dt_col: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out[dt_col] >= start]
    if end is not None:
        out = out[out[dt_col] <= end]
    return out


def setup_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", rotation=0)


# -----------------------------
# Plotting
# -----------------------------
def plot_group(in_dir: str, master: pd.DataFrame, start, end, save_dir: str, fmt: str,
               label_mode: str, mask_threshold: float, ymin=None, ymax=None,
               station_ids: set[str] | None = None):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # map station_id -> metadata
    meta = master.set_index("station_id").to_dict(orient="index")

    station_paths = sorted(glob.glob(os.path.join(in_dir, "station_*.csv")))
    if not station_paths:
        raise FileNotFoundError(f"No station_*.csv found in {in_dir}")
    if station_ids:
        station_paths = [
            p for p in station_paths
            if Path(p).stem.replace("station_", "") in station_ids
        ]

    # group by river_group using master csv metadata
    groups: dict[str, list[str]] = {}
    for p in station_paths:
        sid = Path(p).stem.replace("station_", "")
        info = meta.get(sid, {})
        g = info.get("river_group", "UNKNOWN")
        groups.setdefault(g, []).append(p)

    for gname, paths in groups.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        any_plotted = False
        for p in paths:
            df, dt_col, q_col = load_one_station_csv(p)
            # override threshold if user wants different
            df[q_col] = pd.to_numeric(df[q_col], errors="coerce")
            df.loc[df[q_col] <= mask_threshold, q_col] = np.nan

            dfw = apply_time_window(df, dt_col, start, end)
            if dfw.empty:
                continue

            label = choose_label(df, label_mode)
            ax.plot(dfw[dt_col], dfw[q_col], label=label)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_title(f"river_group = {gname} (discharge)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Discharge (m³/s)")
        ax.grid(True, alpha=0.3)

        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        setup_time_axis(ax)

        # Legend outside if many stations
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"river_group__{gname}.{fmt}")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_station(in_dir: str, start, end, save_dir: str, fmt: str,
                 label_mode: str, mask_threshold: float, ymin=None, ymax=None,
                 station_ids: set[str] | None = None):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    station_paths = sorted(glob.glob(os.path.join(in_dir, "station_*.csv")))
    if not station_paths:
        raise FileNotFoundError(f"No station_*.csv found in {in_dir}")
    if station_ids:
        station_paths = [
            p for p in station_paths
            if Path(p).stem.replace("station_", "") in station_ids
        ]

    for p in station_paths:
        df, dt_col, q_col = load_one_station_csv(p)
        df.loc[df[q_col] <= mask_threshold, q_col] = np.nan

        dfw = apply_time_window(df, dt_col, start, end)
        if dfw.empty:
            continue

        label = choose_label(df, label_mode)
        sid = str(df["station_id"].iloc[0]) if "station_id" in df.columns else Path(p).stem.replace("station_", "")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dfw[dt_col], dfw[q_col], label=label)
        ax.set_title(f"station = {sid} (discharge)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Discharge (m³/s)")
        ax.grid(True, alpha=0.3)

        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        setup_time_axis(ax)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()

        out_path = os.path.join(save_dir, f"station__{sid}.{fmt}")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot MLIT river discharge station CSVs produced by river_down.py"
    )
    ap.add_argument("--in_dir", "--in-dir", dest="in_dir", required=True,
                    help="Directory containing station_*.csv")
    ap.add_argument("--mode", choices=["group", "station"], default="group",
                    help="group: one figure per river_group; station: one figure per station")
    ap.add_argument("--master_csv", required=False,
                    help="Master station list CSV (needed for --mode group).")
    ap.add_argument("--start", default=None, help="Start date (YYYY-MM-DD). Optional.")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD). Optional.")
    ap.add_argument("--save_dir", "--save-dir", "--out", dest="save_dir", required=True,
                    help="Output directory for figures")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"], help="Figure format")

    ap.add_argument("--label_mode", "--label", dest="label_mode", default="station_id",
                    choices=["station_id", "station_id_name"],
                    help="Legend labels: station_id or station_id:station_name")
    ap.add_argument("--mask_threshold", type=float, default=-9999.0,
                    help="Mask values <= this threshold as missing (default masks -10000 etc.)")
    ap.add_argument("--ymin", type=float, default=None, help="Optional y-axis min")
    ap.add_argument("--ymax", type=float, default=None, help="Optional y-axis max")
    ap.add_argument("--station-list", default=None,
                    help="Comma-separated station IDs to plot (e.g., 303051,303052)")

    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    station_ids = parse_station_list(args.station_list)

    if args.mode == "group":
        if not args.master_csv:
            raise SystemExit("--master_csv is required when --mode group")
        master = load_master_csv(args.master_csv)
        plot_group(
            in_dir=args.in_dir,
            master=master,
            start=start,
            end=end,
            save_dir=args.save_dir,
            fmt=args.fmt,
            label_mode=args.label_mode,
            mask_threshold=args.mask_threshold,
            ymin=args.ymin,
            ymax=args.ymax,
            station_ids=station_ids,
        )
    else:
        plot_station(
            in_dir=args.in_dir,
            start=start,
            end=end,
            save_dir=args.save_dir,
            fmt=args.fmt,
            label_mode=args.label_mode,
            mask_threshold=args.mask_threshold,
            ymin=args.ymin,
            ymax=args.ymax,
            station_ids=station_ids,
        )


if __name__ == "__main__":
    main()
