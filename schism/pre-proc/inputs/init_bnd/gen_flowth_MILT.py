#!/usr/bin/env python3
"""Generate flux.th from MLIT river discharge CSVs with robust gap filling.

This version supports hindcast filling when target-window data are missing by using
statistical transfer from donor rivers plus climatology fallback.
"""

from pathlib import Path

from pylib import *
import numpy as np
import pandas as pd

# -----------------------------
# User config
# -----------------------------
milts = [
    "./data/station_302041282207100.csv",
    "./data/station_302031282207050.csv",
    "./data/station_302031282207120.csv",
    "./data/station_302021282206050.csv",
    "./data/station_302021282224010.csv",
    "./data/station_302011282206060.csv",
]  # CSV columns: datetime, value_cms

# Output window (UTC)
start_utc = "2012-01-01 00:00:00"
end_utc = "2014-12-31 23:00:00"

dt = 3600  # output time step [s]
sname = "flux_MILT.th"  # output file name
pt = 1  # quick plot check: 1=on
lwp = 0
cutfreq = 13  # low-pass cutoff [hours]

tz = "Asia/Tokyo"  # source timezone; set None if already UTC

# Gap-filling controls
short_gap_hours = 12  # interpolation only for gaps up to this duration
n_donors = 3  # max donor rivers used in hindcast blend
min_overlap_hours = 24 * 30  # minimum overlap to calibrate donor model
min_points_month = 24 * 5  # minimum monthly points for monthly regression
min_corr = 0.2  # minimum donor correlation for use
use_log_transform = True  # fit on log1p(flow) for stability
climatology_stat = "median"  # median | mean
fill_negative_to_zero = True

# Diagnostics
qc_csv = "flux_MILT_qc.csv"
save_filled_csv = 0
filled_csv_dir = "filled_river_csv"


def _to_utc_datetime(dt_series: pd.Series, src_tz: str | None) -> pd.Series:
    t = pd.to_datetime(dt_series, errors="coerce")
    if src_tz is None:
        if t.dt.tz is None:
            return t.dt.tz_localize("UTC")
        return t.dt.tz_convert("UTC")

    if t.dt.tz is None:
        return t.dt.tz_localize(src_tz, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    return t.dt.tz_convert("UTC")


def _read_station_series(path: str, src_tz: str | None) -> pd.Series:
    df = pd.read_csv(path, usecols=["datetime", "value_cms"], low_memory=False)
    if "datetime" not in df.columns:
        raise ValueError(f"Missing datetime column in {path}")
    if "value_cms" not in df.columns:
        raise ValueError(f"Missing value_cms column in {path}")

    t_utc = _to_utc_datetime(df["datetime"], src_tz)
    q = pd.to_numeric(df["value_cms"], errors="coerce")

    s = pd.Series(q.values, index=t_utc)
    s = s[~s.index.isna()]
    s = s.groupby(level=0).mean().sort_index()
    if fill_negative_to_zero:
        s[s < 0] = np.nan
    return s.astype(float)


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if x.size < 2:
        return None
    xmean = x.mean()
    ymean = y.mean()
    denom = np.sum((x - xmean) ** 2)
    if denom <= 0:
        return None
    b = np.sum((x - xmean) * (y - ymean)) / denom
    a = ymean - b * xmean
    return float(a), float(b)


def _fit_donor_model(
    target: pd.Series,
    donor: pd.Series,
    min_overlap: int,
    min_month_pts: int,
    use_log: bool,
):
    pair = pd.concat([target.rename("t"), donor.rename("d")], axis=1).dropna()
    if len(pair) < min_overlap:
        return None

    x = pair["d"].values
    y = pair["t"].values
    if use_log:
        x = np.log1p(np.clip(x, 0.0, None))
        y = np.log1p(np.clip(y, 0.0, None))

    global_ab = _fit_linear(x, y)
    if global_ab is None:
        return None

    corr = np.corrcoef(pair["d"].values, pair["t"].values)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0

    month_ab = {}
    for m in range(1, 13):
        pm = pair[pair.index.month == m]
        if len(pm) < min_month_pts:
            continue
        xm = pm["d"].values
        ym = pm["t"].values
        if use_log:
            xm = np.log1p(np.clip(xm, 0.0, None))
            ym = np.log1p(np.clip(ym, 0.0, None))
        ab = _fit_linear(xm, ym)
        if ab is not None:
            month_ab[m] = ab

    return {
        "corr": float(corr),
        "global_ab": global_ab,
        "month_ab": month_ab,
        "use_log": use_log,
    }


def _predict_from_model(donor_vals: np.ndarray, months: np.ndarray, model: dict) -> np.ndarray:
    pred = np.full(donor_vals.shape, np.nan, dtype=float)
    valid = np.isfinite(donor_vals)
    if not np.any(valid):
        return pred

    x = donor_vals.copy()
    if model["use_log"]:
        x = np.log1p(np.clip(x, 0.0, None))

    for m in range(1, 13):
        fpt = valid & (months == m)
        if not np.any(fpt):
            continue
        ab = model["month_ab"].get(m, model["global_ab"])
        a, b = ab
        yhat = a + b * x[fpt]
        if model["use_log"]:
            yhat = np.expm1(yhat)
        pred[fpt] = yhat

    pred[pred < 0] = 0.0
    return pred


def _fill_with_donors(
    target_name: str,
    target_full: pd.Series,
    target_window: pd.Series,
    all_series: dict[str, pd.Series],
    n_max: int,
    min_overlap: int,
    min_month_pts: int,
    min_corr_val: float,
    use_log: bool,
):
    models = []
    months = target_window.index.month.values

    for donor_name, donor_series in all_series.items():
        if donor_name == target_name:
            continue
        model = _fit_donor_model(target_full, donor_series, min_overlap, min_month_pts, use_log)
        if model is None:
            continue
        if model["corr"] < min_corr_val:
            continue
        models.append((donor_name, model))

    models.sort(key=lambda x: x[1]["corr"], reverse=True)
    models = models[:n_max]

    if len(models) == 0:
        return np.full(len(target_window), np.nan), []

    preds = []
    weights = []
    used = []
    for donor_name, model in models:
        donor_win = all_series[donor_name].reindex(target_window.index)
        yhat = _predict_from_model(donor_win.values.astype(float), months, model)
        preds.append(yhat)
        # Weight by skill and overlap strength; keep positive weights only.
        wt = model["corr"] if model["corr"] > 0.01 else 0.01
        weights.append(wt)
        used.append((donor_name, model["corr"]))

    preds = np.asarray(preds)
    weights = np.asarray(weights)[:, None]

    valid = np.isfinite(preds)
    wsum = np.sum(np.where(valid, weights, 0.0), axis=0)
    num = np.sum(np.where(valid, preds * weights, 0.0), axis=0)

    blend = np.full(target_window.shape, np.nan, dtype=float)
    ok = wsum > 0
    blend[ok] = num[ok] / wsum[ok]
    blend[blend < 0] = 0.0
    return blend, used


def _monthly_climatology(series: pd.Series, target_index: pd.DatetimeIndex, stat_name: str) -> np.ndarray:
    valid = series.dropna()
    if len(valid) == 0:
        return np.zeros(len(target_index), dtype=float)

    if stat_name == "mean":
        monthly = valid.groupby(valid.index.month).mean()
        fallback = float(valid.mean())
    else:
        monthly = valid.groupby(valid.index.month).median()
        fallback = float(valid.median())

    out = np.full(len(target_index), np.nan, dtype=float)
    months = target_index.month
    for m in range(1, 13):
        fpt = months == m
        if not np.any(fpt):
            continue
        out[fpt] = float(monthly.get(m, fallback))
    out[~np.isfinite(out)] = fallback
    out[out < 0] = 0.0
    return out


def _build_filled_series(name: str, all_series: dict[str, pd.Series], target_index: pd.DatetimeIndex):
    source = all_series[name].reindex(target_index)
    filled = source.copy()

    qc_flag = np.full(len(target_index), -1, dtype=int)
    qc_flag[np.isfinite(source.values)] = 0  # observed

    max_interp_steps = int(round(short_gap_hours * 3600 / dt))
    if max_interp_steps > 0:
        interp = filled.interpolate(method="time", limit=max_interp_steps, limit_area="inside")
        new_interp = interp.notna() & filled.isna()
        filled = interp
        qc_flag[new_interp.values] = 1

    miss = filled.isna().values
    donor_fill = np.full(len(target_index), np.nan, dtype=float)
    donor_used = []
    if np.any(miss):
        donor_fill, donor_used = _fill_with_donors(
            name,
            all_series[name],
            filled,
            all_series,
            n_donors,
            min_overlap_hours,
            min_points_month,
            min_corr,
            use_log_transform,
        )
        fillable = miss & np.isfinite(donor_fill)
        if np.any(fillable):
            arr = filled.values.astype(float)
            arr[fillable] = donor_fill[fillable]
            filled = pd.Series(arr, index=target_index)
            qc_flag[fillable] = 2

    miss = filled.isna().values
    if np.any(miss):
        clim = _monthly_climatology(all_series[name], target_index, climatology_stat)
        fillable = miss & np.isfinite(clim)
        if np.any(fillable):
            arr = filled.values.astype(float)
            arr[fillable] = clim[fillable]
            filled = pd.Series(arr, index=target_index)
            qc_flag[fillable] = 3

    miss = filled.isna().values
    if np.any(miss):
        arr = filled.values.astype(float)
        arr[miss] = 0.0
        filled = pd.Series(arr, index=target_index)
        qc_flag[miss] = 4

    if fill_negative_to_zero:
        arr = filled.values.astype(float)
        arr[arr < 0] = 0.0
        filled = pd.Series(arr, index=target_index)

    return filled, qc_flag, donor_used


def _to_datenum(ts_utc: pd.Timestamp) -> float:
    # ts_utc should represent UTC clock time.
    return float(datenum(ts_utc.year, ts_utc.month, ts_utc.day, ts_utc.hour, ts_utc.minute, ts_utc.second))


# -----------------------------
# Build flux.th
# -----------------------------
start_ts = pd.Timestamp(start_utc, tz="UTC")
end_ts = pd.Timestamp(end_utc, tz="UTC")
if end_ts < start_ts:
    raise ValueError("end_utc must be later than start_utc")

# datenum-based elapsed time vector for SCHISM flux.th
st = _to_datenum(start_ts)
et = _to_datenum(end_ts)
ntime = np.arange(0.0, (et - st) * 86400.0 + 0.1 * dt, dt, dtype=float)
target_index = pd.date_range(start=start_ts, periods=len(ntime), freq=f"{int(dt)}s")

series_all = {}
for file in milts:
    key = Path(file).stem
    s = _read_station_series(file, tz)
    if len(s) == 0:
        raise ValueError(f"No valid rows found in {file}")
    series_all[key] = s
    print(f"Loaded {key}: {s.index.min()} to {s.index.max()} ({len(s)} rows)")

newset = ntime.copy()
qc_rows = []

if save_filled_csv:
    Path(filled_csv_dir).mkdir(parents=True, exist_ok=True)

for i, key in enumerate(series_all.keys(), start=1):
    filled, qc_flag, donors = _build_filled_series(key, series_all, target_index)
    nrd = -filled.values.astype(float)

    if lwp == 1:
        nrd = lpfilt(nrd, dt / 3600.0 / 24.0, cutfreq / 24.0)

    newset = column_stack((newset, nrd))

    counts = {
        "station": key,
        "obs": int(np.sum(qc_flag == 0)),
        "interp": int(np.sum(qc_flag == 1)),
        "donor": int(np.sum(qc_flag == 2)),
        "clim": int(np.sum(qc_flag == 3)),
        "zero": int(np.sum(qc_flag == 4)),
        "total": int(len(qc_flag)),
        "donors": ";".join([f"{n}:{c:.3f}" for n, c in donors]),
    }
    qc_rows.append(counts)
    print(
        f"[{i}/{len(series_all)}] {key}: "
        f"obs={counts['obs']}, interp={counts['interp']}, donor={counts['donor']}, "
        f"clim={counts['clim']}, zero={counts['zero']}"
    )

    if save_filled_csv:
        out = pd.DataFrame(
            {
                "datetime_utc": target_index.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"),
                "flow_cms_filled": filled.values,
                "qc_flag": qc_flag,
            }
        )
        out.to_csv(Path(filled_csv_dir) / f"{key}_filled.csv", index=False)

np.savetxt(sname, newset, fmt="%.6f")
print(f"Saved {sname} with shape {newset.shape}")

pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)
print(f"Saved QC summary to {qc_csv}")

if pt == 1:
    fs = loadtxt(sname)
    for nn in arange(shape(fs)[1] - 1):
        plot(fs[:, 0], fs[:, nn + 1])
    xlabel("time (s)")
    ylabel("River discharge (m^3/s); negative in flux.th")
    show()
