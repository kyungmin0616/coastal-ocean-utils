#!/usr/bin/env python3
# generate flux.th using MLIT river discharge (MILT) station CSVs

from pylib import *
import pandas as pd
import numpy as np

# -----------------------------
# User config
# -----------------------------
milts = [
    "/S/data00/G6008/d1041/dataset/MILT/River/out_mlit/station_302011282206030.csv",
]  # MLIT station CSVs with columns: datetime, value_cms
st = datenum(2011, 1, 1)  # start time (UTC) for output
et = datenum(2011, 12, 31)  # end time (UTC) for output
sname = "flux_MILT.th"  # output file name
dt = 900  # time step of flux.th (seconds)
pt = 1  # quick plot check: 1=on
lwp = 1
cutfreq = 13  # low-pass cutoff (hours)

tz = "Asia/Tokyo"  # set to None if datetimes are already in UTC

# -----------------------------
# Build flux.th
# -----------------------------
ntime = arange(0, (et - st) * 86400, dt)
newset = ntime.copy()

for file in milts:
    df = pd.read_csv(file)
    if "datetime" not in df.columns:
        raise ValueError(f"Missing datetime column in {file}")
    if "value_cms" not in df.columns:
        raise ValueError(f"Missing value_cms column in {file}")

    t = pd.to_datetime(df["datetime"], errors="coerce")
    if tz:
        t = t.dt.tz_localize(tz).dt.tz_convert("UTC")
    time = datenum(t.values.astype("str")).astype("float")
    rd = pd.to_numeric(df["value_cms"], errors="coerce").values.astype("float")

    fpt = (time >= st) * (time <= et)
    time = time[fpt]
    rd = rd[fpt]
    if len(time) == 0:
        raise ValueError(f"No data in requested window for {file}")

    time = (time - st) * 86400
    time, idx = unique(time, return_index=True)
    rd = rd[idx]

    nrd = -interpolate.interp1d(time, rd, bounds_error=False, fill_value="extrapolate")(ntime)
    if lwp == 1:
        nrd = lpfilt(nrd, dt / 3600 / 24, cutfreq / 24)
    newset = column_stack((newset, nrd))

np.savetxt(sname, newset, fmt="%f")

if pt == 1:
    fs = loadtxt(sname)
    for nn in arange(shape(fs)[1] - 1):
        plot(fs[:, 0], fs[:, nn + 1])
    xlabel("time (s)")
    ylabel("River discharge (m^3/s)")
    show()
