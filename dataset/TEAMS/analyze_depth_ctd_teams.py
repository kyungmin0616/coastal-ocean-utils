#!/usr/bin/env python3
"""
Compare max CTD observation depth to nearest SCHISM grid depth.

Uses pylib:
  gd = read('hgrid.gr3')
  S  = read('onagawa_d2_ctd.npz')
  sind = near_pts(obs_xy, grid_xy)

Outputs a CSV summary with station name/ID, obs max depth, grid depth, and diff.
"""
import csv
import os

import numpy as np

from pylib import read, near_pts

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


CONFIG = {
    "GRID_PATH": "/Users/kpark/Codes/D26-017-selected/hgrid.gr3",
    "OBS_NPZ": "/Users/kpark/Codes/D26-017-selected/npz/onagawa_d2_ctd.npz",
    "OUT_CSV": "/Users/kpark/Codes/D26-017-selected/ctd_depth_compare.csv",
    "PLOT_DIR": "/Users/kpark/Codes/D26-017-selected/plots_depth_compare",
    "PLOT_MAP": True,
    "PLOT_SCATTER": True,
    "PLOT_BAR": True,
    "MAP_FIGSIZE": (7, 7),
    "MAP_LIMS": (141.4127, 141.6027, 38.3298, 38.49),  # (xmin, xmax, ymin, ymax)
    "MAP_LABEL_FONTSIZE": 7,
    "MAP_LABEL_OFFSET": (3, 3),  # offset in points (x, y)
    "SCATTER_FIGSIZE": (6, 6),
    "SCATTER_LABEL_FONTSIZE": 7,
    "BAR_FIGSIZE": (10, 4),
    "BAR_LABEL_FONTSIZE": 7,
    "BAR_LABEL_EVERY": 1,
    "LON_SYSTEM": "auto",  # auto, 180, 360
    "LABEL_BY": "station_id",  # station_id, station_name, auto
}


def _as_array(val):
    return np.array(val)


def _to_float_array(val):
    return _as_array(val).astype(float)


def _wrap_lon(lon, system):
    if system == "360":
        return lon % 360.0
    if system == "180":
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _align_lon(obs_lon, grid_lon, system):
    if system in ("180", "360"):
        return _wrap_lon(obs_lon, system)
    # auto: try to match grid range
    if np.nanmax(grid_lon) <= 180.0 and np.nanmax(obs_lon) > 180.0:
        return _wrap_lon(obs_lon, "180")
    if np.nanmin(grid_lon) >= 0.0 and np.nanmin(obs_lon) < 0.0:
        return _wrap_lon(obs_lon, "360")
    return obs_lon


def _station_label(idx, station_name, station_id, mode):
    name = ""
    if mode == "station_name":
        if station_name is not None and station_name[idx]:
            name = station_name[idx]
    elif mode == "station_id":
        if station_id is not None and station_id[idx]:
            name = station_id[idx]
    else:
        if station_name is not None and station_name[idx]:
            name = station_name[idx]
        elif station_id is not None and station_id[idx]:
            name = station_id[idx]
    if not name:
        return f"pt_{idx:05d}"
    return str(name)


def _approx_distance_km(lon1, lat1, lon2, lat2):
    deg2km = 111.0
    latm = 0.5 * (lat1 + lat2)
    dx = (lon2 - lon1) * np.cos(np.deg2rad(latm)) * deg2km
    dy = (lat2 - lat1) * deg2km
    return float(np.sqrt(dx * dx + dy * dy))


def _plot_map(gd, obs_xy, grid_xy, stations, out_dir):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=CONFIG["MAP_FIGSIZE"])
    gd.plot_bnd()
    ax = plt.gca()
    ax.scatter(obs_xy[:, 0], obs_xy[:, 1], s=18, c="tab:blue", label="CTD obs", zorder=3)
    ax.scatter(grid_xy[:, 0], grid_xy[:, 1], s=14, c="tab:red", marker="x", label="Nearest grid", zorder=3)
    dx, dy = CONFIG["MAP_LABEL_OFFSET"]
    for i, station in enumerate(stations):
        ax.annotate(
            str(station),
            xy=(obs_xy[i, 0], obs_xy[i, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=CONFIG["MAP_LABEL_FONTSIZE"],
            ha="left",
            va="bottom",
            color="tab:blue",
            clip_on=True,
        )
    ax.legend(loc="best")
    ax.set_title("CTD locations and nearest grid nodes")
    if CONFIG["MAP_LIMS"]:
        xmin, xmax, ymin, ymax = CONFIG["MAP_LIMS"]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    fig_path = os.path.join(out_dir, "ctd_grid_map.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_scatter(obs_depths, grid_depths, stations, out_dir):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=CONFIG["SCATTER_FIGSIZE"])
    ax.scatter(grid_depths, obs_depths, s=25, alpha=0.8)
    for i, station in enumerate(stations):
        ax.text(
            grid_depths[i],
            obs_depths[i],
            str(station),
            fontsize=CONFIG["SCATTER_LABEL_FONTSIZE"],
            ha="left",
            va="bottom",
        )
    vmin = min(np.nanmin(grid_depths), np.nanmin(obs_depths))
    vmax = max(np.nanmax(grid_depths), np.nanmax(obs_depths))
    ax.plot([vmin, vmax], [vmin, vmax], "--", color="0.4", lw=1)
    ax.set_xlabel("Grid depth (m)")
    ax.set_ylabel("Obs max depth (m)")
    ax.set_title("Obs vs grid depth (1:1 line)")
    ax.grid(True, alpha=0.3)
    fig_path = os.path.join(out_dir, "ctd_depth_scatter.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_bar(stations, diffs, out_dir):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    os.makedirs(out_dir, exist_ok=True)
    order = np.argsort(np.abs(diffs))[::-1]
    diffs = diffs[order]
    stations = [stations[i] for i in order]
    fig, ax = plt.subplots(figsize=CONFIG["BAR_FIGSIZE"])
    ax.bar(np.arange(len(diffs)), diffs, color="tab:gray")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("Obs - grid depth (m)")
    ax.set_title("Depth difference by station (sorted by |diff|)")
    step = max(1, int(CONFIG["BAR_LABEL_EVERY"]))
    idxs = np.arange(len(stations))[::step]
    ax.set_xticks(idxs)
    ax.set_xticklabels([stations[i] for i in idxs], rotation=90, fontsize=CONFIG["BAR_LABEL_FONTSIZE"])
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "ctd_depth_diff_bar.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main():
    if not os.path.exists(CONFIG["GRID_PATH"]):
        raise SystemExit(f"Missing grid: {CONFIG['GRID_PATH']}")
    if not os.path.exists(CONFIG["OBS_NPZ"]):
        raise SystemExit(f"Missing observations: {CONFIG['OBS_NPZ']}")

    gd = read(CONFIG["GRID_PATH"])
    S = read(CONFIG["OBS_NPZ"])

    obs_lon = _to_float_array(S.lon)
    obs_lat = _to_float_array(S.lat)
    obs_depth = _to_float_array(S.depth)

    station_name = _as_array(getattr(S, "station_name", []))
    station_id = _as_array(getattr(S, "station_id", []))

    grid_lon = _to_float_array(gd.x)
    grid_lat = _to_float_array(gd.y)
    grid_dp = _to_float_array(gd.dp)

    obs_lon = _align_lon(obs_lon, grid_lon, CONFIG["LON_SYSTEM"])

    valid = np.isfinite(obs_lon) & np.isfinite(obs_lat) & np.isfinite(obs_depth)
    if not np.any(valid):
        raise SystemExit("No valid observations with lon/lat/depth.")

    idxs = np.where(valid)[0]
    groups = {}
    label_mode = CONFIG.get("LABEL_BY", "auto")
    for i in idxs:
        key = _station_label(i, station_name, station_id, label_mode)
        rec = groups.get(key)
        if rec is None:
            groups[key] = dict(lon=[], lat=[], depth=[])
        groups[key]["lon"].append(obs_lon[i])
        groups[key]["lat"].append(obs_lat[i])
        groups[key]["depth"].append(obs_depth[i])

    keys = sorted(groups.keys())
    obs_xy = []
    max_depths = []
    for key in keys:
        lons = np.array(groups[key]["lon"], dtype=float)
        lats = np.array(groups[key]["lat"], dtype=float)
        depths = np.array(groups[key]["depth"], dtype=float)
        obs_xy.append([np.nanmean(lons), np.nanmean(lats)])
        max_depths.append(np.nanmax(depths))

    obs_xy = np.array(obs_xy, dtype=float)
    max_depths = np.array(max_depths, dtype=float)

    grid_xy = np.c_[grid_lon, grid_lat]
    sind = np.array(near_pts(obs_xy, grid_xy)).astype(int).ravel()
    grid_depths = grid_dp[sind]
    grid_xy_near = grid_xy[sind]

    rows = []
    for i, key in enumerate(keys):
        lon0, lat0 = obs_xy[i]
        gidx = int(sind[i])
        glon = float(grid_lon[gidx])
        glat = float(grid_lat[gidx])
        gdp = float(grid_depths[i])
        odep = float(max_depths[i])
        diff = odep - gdp
        dist_km = _approx_distance_km(lon0, lat0, glon, glat)
        rows.append(
            {
                "station": key,
                "obs_lon": lon0,
                "obs_lat": lat0,
                "obs_max_depth": odep,
                "grid_node": gidx + 1,
                "grid_lon": glon,
                "grid_lat": glat,
                "grid_depth": gdp,
                "obs_minus_grid": diff,
                "distance_km": dist_km,
            }
        )

    out_csv = CONFIG["OUT_CSV"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "station",
                "obs_lon",
                "obs_lat",
                "obs_max_depth",
                "grid_node",
                "grid_lon",
                "grid_lat",
                "grid_depth",
                "obs_minus_grid",
                "distance_km",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} station rows to {out_csv}")

    if CONFIG["PLOT_MAP"]:
        _plot_map(gd, obs_xy, grid_xy_near, keys, CONFIG["PLOT_DIR"])
    if CONFIG["PLOT_SCATTER"]:
        _plot_scatter(max_depths, grid_depths, keys, CONFIG["PLOT_DIR"])
    if CONFIG["PLOT_BAR"]:
        _plot_bar(keys, max_depths - grid_depths, CONFIG["PLOT_DIR"])


if __name__ == "__main__":
    main()
