#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ----------------------------
# Config
# ----------------------------
INPUT_PATH = (
    "/Users/kpark/Documents/DEM/m7000/M7005/ascii/M7005_SanrikuOffshore_Ver.2.3"
)
OUTPUT_PATH = "/Users/kpark/Downloads/M7005_三陸沖_Ver.2.3.npz"
MODE = "auto"  # auto | grid | point
GRID_DECIMALS = 5  # rounding for grid detection
COMPRESS = True
SAVE_GROUP = True
SAVE_REC_TYPE = True
FLIP_DEPTH = False
GROUP_REPORT = True
GROUP_REPORT_OUT = "/Users/kpark/Downloads/m7000_group_resolution.csv"
GROUP_SPACING_STAT = "mean"  # mean | median
REPORT_MEAN = True
REPORT_MEDIAN = True
PLOT = True
PLOT_KIND = "scatter"  # auto | contour | scatter
CONTOUR_LEVELS = 40
BATHY_CBAR_LIMS = (0,50)  # (vmin, vmax) or None
PLOT_RESOLUTION = True
RES_PLOT_KIND = "Scatter"  # auto | contour | scatter
RES_CONTOUR_LEVELS = 30
RES_CBAR_LIMS = (0,100)  # (vmin, vmax) or None
GR3_PATH = "/Users/kpark/Downloads/01.gr3"
PLOT_OUT = "m7000_bathy.png"
RES_PLOT_OUT = "m7000_resolution.png"
PLOT_SHOW = True
PLOT_MAX_POINTS = 200000
PLOT_ALPHA = 0.7
PLOT_SIZE = 2.0


def _read_m7000(path: Path):
    rec_type = []
    depth = []
    lat = []
    lon = []
    group = []
    meta = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rec_id = parts[0]
            rec_type.append(rec_id[0])
            depth.append(float(parts[1]))
            lat.append(float(parts[2]))
            lon.append(float(parts[3]))
            if SAVE_GROUP:
                group.append(int(parts[5]))
            if meta is None:
                meta = parts[6]

    depth = np.asarray(depth, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    rec_type = np.asarray(rec_type, dtype="U1")
    group = np.asarray(group, dtype=np.int32) if SAVE_GROUP else None

    return depth, lat, lon, rec_type, group, meta


def _as_grid(depth, lat, lon, decimals):
    lat_r = np.round(lat, decimals=decimals)
    lon_r = np.round(lon, decimals=decimals)
    uniq_lat = np.unique(lat_r)
    uniq_lon = np.unique(lon_r)

    if uniq_lat.size * uniq_lon.size != lat.size:
        return None

    lat_idx = np.searchsorted(uniq_lat, lat_r)
    lon_idx = np.searchsorted(uniq_lon, lon_r)
    flat_idx = lat_idx * uniq_lon.size + lon_idx
    if np.unique(flat_idx).size != lat.size:
        return None

    grid = np.full((uniq_lat.size, uniq_lon.size), np.nan, dtype=np.float32)
    grid[lat_idx, lon_idx] = depth
    return uniq_lat, uniq_lon, grid


def _save_npz(path, payload, compress):
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371000.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r * c


def _group_spacing_stats(lat, lon, group):
    same = group[1:] == group[:-1]
    if not np.any(same):
        return []

    lat1 = lat[:-1][same]
    lon1 = lon[:-1][same]
    lat2 = lat[1:][same]
    lon2 = lon[1:][same]
    g = group[1:][same]

    dist = _haversine(lat1, lon1, lat2, lon2)

    order = np.argsort(g)
    g_sorted = g[order]
    dist_sorted = dist[order]
    uniq_g, idx, counts = np.unique(g_sorted, return_index=True, return_counts=True)

    uniq_all, cnt_all = np.unique(group, return_counts=True)
    n_points = dict(zip(uniq_all.tolist(), cnt_all.tolist()))

    rows = []
    for gid, start, count in zip(uniq_g, idx, counts):
        seg = dist_sorted[start:start + count]
        mean_val = float(seg.mean())
        median_val = float(np.median(seg))
        steps = int(count)
        points = int(n_points.get(int(gid), steps + 1))
        rows.append((int(gid), points, steps, mean_val, median_val))
    return rows


def _write_group_report(rows, path, report_mean, report_median):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        headers = ["group_id", "n_points", "n_steps"]
        if report_mean:
            headers.append("mean_spacing_m")
        if report_median:
            headers.append("median_spacing_m")
        f.write(",".join(headers) + "\n")
        for row in rows:
            cols = [str(row[0]), str(row[1]), str(row[2])]
            if report_mean:
                cols.append(f"{row[3]:.3f}")
            if report_median:
                cols.append(f"{row[4]:.3f}")
            f.write(",".join(cols) + "\n")
    print("Saved group resolution report to", path)


def _group_value_map(group, rows, stat):
    if not rows:
        return None
    uniq_all = np.unique(group)
    values = np.full(uniq_all.shape, np.nan, dtype=np.float64)
    stat_vals = {
        row[0]: (row[3] if stat == "mean" else row[4]) for row in rows
    }
    ids = np.fromiter(stat_vals.keys(), dtype=np.int32, count=len(stat_vals))
    vals = np.fromiter(stat_vals.values(), dtype=np.float64, count=len(stat_vals))
    pos = np.searchsorted(uniq_all, ids)
    values[pos] = vals
    idx = np.searchsorted(uniq_all, group)
    return values[idx]


def _levels_from_lims(cbar_lims, n_levels):
    if cbar_lims is None:
        return n_levels
    vmin, vmax = cbar_lims
    return np.linspace(vmin, vmax, n_levels)


def _plot_bathy(
    lat,
    lon,
    elev,
    gr3_path,
    plot_out,
    show,
    max_points,
    alpha,
    size,
    plot_kind,
    contour_levels,
    cbar_lims,
    title,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip plot.")
        return
    try:
        from pylib import read
    except ImportError:
        print("pylib not available; skip plot.")
        return

    gd = read(gr3_path)

    if elev.ndim == 2 and lat.ndim == 1 and lon.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
        lonp = lon2.ravel()
        latp = lat2.ravel()
        elevp = elev.ravel()
        grid_mode = True
    else:
        lonp = lon
        latp = lat
        elevp = elev
        grid_mode = False

    mask = np.isfinite(elevp)
    lonp = lonp[mask]
    latp = latp[mask]
    elevp = elevp[mask]

    if lonp.size > max_points:
        step = max(1, lonp.size // max_points)
        lonp = lonp[::step]
        latp = latp[::step]
        elevp = elevp[::step]
        print(f"Plot decimated to ~{lonp.size} points (step={step}).")

    fig, ax = plt.subplots(figsize=(8, 7))
    gd.plot_bnd()
    ax = plt.gca()

    use_contour = plot_kind in ("auto", "contour")
    artist = None
    vmin = vmax = None
    levels = _levels_from_lims(cbar_lims, contour_levels)
    if cbar_lims is not None:
        vmin, vmax = cbar_lims

    if use_contour and grid_mode:
        artist = ax.contourf(lon, lat, elev, levels=levels, cmap="viridis")
    elif use_contour and lonp.size >= 3:
        try:
            artist = ax.tricontourf(
                lonp, latp, elevp, levels=levels, cmap="viridis"
            )
        except Exception as exc:
            print("tricontourf failed; fallback to scatter:", exc)
            artist = ax.scatter(
                lonp, latp, c=elevp, s=size, alpha=alpha, cmap="viridis",
                vmin=vmin, vmax=vmax,
            )
    else:
        artist = ax.scatter(
            lonp, latp, c=elevp, s=size, alpha=alpha, cmap="viridis",
            vmin=vmin, vmax=vmax,
        )

    cbar = fig.colorbar(artist, ax=ax)
    cbar.set_label("Depth (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.tight_layout()

    if plot_out:
        fig.savefig(plot_out, dpi=150)
        print("Saved plot to", plot_out)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Convert JHA M7000 ASCII bathymetry to NPZ."
    )
    parser.add_argument("--input", default=INPUT_PATH, help="Input ASCII file")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output NPZ file")
    parser.add_argument(
        "--mode",
        default=MODE,
        choices=["auto", "grid", "point"],
        help="Output mode",
    )
    parser.add_argument(
        "--grid-decimals",
        default=GRID_DECIMALS,
        type=int,
        help="Rounding decimals for grid detection",
    )
    parser.add_argument(
        "--compress",
        default=COMPRESS,
        action=argparse.BooleanOptionalAction,
        help="Use np.savez_compressed",
    )
    parser.add_argument(
        "--flip-depth",
        default=FLIP_DEPTH,
        action=argparse.BooleanOptionalAction,
        help="Flip depth sign",
    )
    parser.add_argument(
        "--group-report",
        default=GROUP_REPORT,
        action=argparse.BooleanOptionalAction,
        help="Report representative spacing per group",
    )
    parser.add_argument(
        "--group-report-out",
        default=GROUP_REPORT_OUT,
        help="Group spacing report CSV",
    )
    parser.add_argument(
        "--group-spacing-stat",
        default=GROUP_SPACING_STAT,
        choices=["mean", "median"],
        help="Spacing statistic for representative resolution map",
    )
    parser.add_argument(
        "--plot",
        default=PLOT,
        action=argparse.BooleanOptionalAction,
        help="Plot bathymetry with gd.plot_bnd()",
    )
    parser.add_argument(
        "--plot-kind",
        default=PLOT_KIND,
        choices=["auto", "contour", "scatter"],
        help="Plot style",
    )
    parser.add_argument(
        "--contour-levels",
        default=CONTOUR_LEVELS,
        type=int,
        help="Number of contour levels",
    )
    parser.add_argument(
        "--bathy-cbar-lims",
        nargs=2,
        type=float,
        default=BATHY_CBAR_LIMS,
        metavar=("VMIN", "VMAX"),
        help="Colorbar limits for bathymetry",
    )
    parser.add_argument(
        "--plot-resolution",
        default=PLOT_RESOLUTION,
        action=argparse.BooleanOptionalAction,
        help="Plot representative resolution map",
    )
    parser.add_argument(
        "--res-plot-kind",
        default=RES_PLOT_KIND,
        choices=["auto", "contour", "scatter"],
        help="Plot style for resolution map",
    )
    parser.add_argument(
        "--res-contour-levels",
        default=RES_CONTOUR_LEVELS,
        type=int,
        help="Number of contour levels for resolution map",
    )
    parser.add_argument(
        "--res-cbar-lims",
        nargs=2,
        type=float,
        default=RES_CBAR_LIMS,
        metavar=("VMIN", "VMAX"),
        help="Colorbar limits for resolution map",
    )
    parser.add_argument("--gr3", default=GR3_PATH, help="GR3 grid for boundary plot")
    parser.add_argument("--plot-out", default=PLOT_OUT, help="Plot output file")
    parser.add_argument(
        "--res-plot-out",
        default=RES_PLOT_OUT,
        help="Resolution plot output file",
    )
    parser.add_argument(
        "--show",
        default=PLOT_SHOW,
        action=argparse.BooleanOptionalAction,
        help="Show plot interactively",
    )
    parser.add_argument(
        "--plot-max-points",
        default=PLOT_MAX_POINTS,
        type=int,
        help="Max points to plot (decimate if larger)",
    )
    parser.add_argument(
        "--plot-alpha",
        default=PLOT_ALPHA,
        type=float,
        help="Scatter alpha",
    )
    parser.add_argument(
        "--plot-size",
        default=PLOT_SIZE,
        type=float,
        help="Scatter marker size",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    depth, lat, lon, rec_type, group, meta = _read_m7000(in_path)
    if args.flip_depth:
        depth = -depth

    print(
        "Loaded", len(depth), "points; depth min/max:", float(depth.min()), float(depth.max())
    )
    print(
        "Lon range:", float(lon.min()), float(lon.max()),
        "Lat range:", float(lat.min()), float(lat.max()),
    )

    spacing_rows = []
    if group is not None and (args.group_report or args.plot_resolution):
        spacing_rows = _group_spacing_stats(lat, lon, group)
        if spacing_rows and args.group_report:
            _write_group_report(spacing_rows, args.group_report_out, REPORT_MEAN, REPORT_MEDIAN)
            vals = np.array([r[3] if args.group_spacing_stat == "mean" else r[4]
                             for r in spacing_rows], dtype=float)
            print(
                f"Group spacing ({args.group_spacing_stat}) (m): count=", len(vals),
                "min=", float(vals.min()),
                "median=", float(np.median(vals)),
                "max=", float(vals.max()),
            )
        elif args.group_report:
            print("No consecutive points found for group spacing report.")

    payload = {"source": str(in_path)}
    if meta is not None:
        payload["meta"] = np.asarray(meta)

    if args.mode in ("auto", "grid"):
        grid = _as_grid(depth, lat, lon, decimals=args.grid_decimals)
        if grid is not None:
            uniq_lat, uniq_lon, elev = grid
            payload.update({
                "lat": uniq_lat,
                "lon": uniq_lon,
                "elev": elev,
            })
            if SAVE_REC_TYPE:
                payload["rec_type"] = rec_type
            if SAVE_GROUP and group is not None:
                payload["group"] = group
            _save_npz(out_path, payload, args.compress)
            print("Saved gridded NPZ to", out_path)
            if args.plot:
                _plot_bathy(
                    payload["lat"],
                    payload["lon"],
                    payload["elev"],
                    args.gr3,
                    args.plot_out,
                    args.show,
                    args.plot_max_points,
                    args.plot_alpha,
                    args.plot_size,
                    args.plot_kind,
                    args.contour_levels,
                    args.bathy_cbar_lims,
                    "M7000 Bathymetry",
                )
            if args.plot_resolution and spacing_rows and group is not None:
                spacing = _group_value_map(group, spacing_rows, args.group_spacing_stat)
                _plot_bathy(
                    lat,
                    lon,
                    spacing,
                    args.gr3,
                    args.res_plot_out,
                    args.show,
                    args.plot_max_points,
                    args.plot_alpha,
                    args.plot_size,
                    args.res_plot_kind,
                    args.res_contour_levels,
                    args.res_cbar_lims,
                    f"M7000 Resolution ({args.group_spacing_stat})",
                )
            return 0
        if args.mode == "grid":
            raise SystemExit("Grid detection failed. Try --mode point.")

    payload.update({
        "lat": lat,
        "lon": lon,
        "elev": depth,
    })
    if SAVE_REC_TYPE:
        payload["rec_type"] = rec_type
    if SAVE_GROUP and group is not None:
        payload["group"] = group

    _save_npz(out_path, payload, args.compress)
    print("Saved point-cloud NPZ to", out_path)
    if args.plot:
        _plot_bathy(
            payload["lat"],
            payload["lon"],
            payload["elev"],
            args.gr3,
            args.plot_out,
            args.show,
            args.plot_max_points,
            args.plot_alpha,
            args.plot_size,
            args.plot_kind,
            args.contour_levels,
            args.bathy_cbar_lims,
            "M7000 Bathymetry",
        )
    if args.plot_resolution and spacing_rows and group is not None:
        spacing = _group_value_map(group, spacing_rows, args.group_spacing_stat)
        _plot_bathy(
            lat,
            lon,
            spacing,
            args.gr3,
            args.res_plot_out,
            args.show,
            args.plot_max_points,
            args.plot_alpha,
            args.plot_size,
            args.res_plot_kind,
            args.res_contour_levels,
            args.res_cbar_lims,
            f"M7000 Resolution ({args.group_spacing_stat})",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
