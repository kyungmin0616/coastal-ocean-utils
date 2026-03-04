#!/usr/bin/env python3
from __future__ import annotations

"""
Convert JHA M7000 ASCII bathymetry to NPZ and apply an approximate vertical-datum
conversion from NLLW (Nearly Lowest Low Water) to Tokyo Peil (TP).

This script uses tide-station offsets (TP - NLLW) and interpolates a spatially
varying offset field over the M7000 domain using IDW (or nearest neighbor).

Important:
- The built-in station offsets are approximate values provided by the user.
- Offshore extrapolation is approximate when only coastal stations are available.
- The conversion assumes the M7000 values are depth (positive down) after any
  optional --flip-depth step.

Depth conversion used (default, depth positive-down):
  depth_TP = depth_NLLW + (TP - NLLW)
"""

import argparse
from pathlib import Path

import numpy as np

from m2npz import _as_grid, _levels_from_lims, _read_m7000, _save_npz

# ----------------------------
# Config (CLI overrides these)
# ----------------------------
INPUT_PATH = "/Users/kpark/Documents/DEM/m7000/M7005/ascii/M7005_SanrikuOffshore_Ver.2.3"
OUTPUT_PATH = "/Users/kpark/Downloads/M7005_三陸沖_Ver.2.3_TP.npz"
MODE = "auto"  # auto | grid | point
GRID_DECIMALS = 5
COMPRESS = True
FLIP_DEPTH = False

# Source/target vertical datum labels for metadata
SOURCE_DATUM = "NLLW"
TARGET_DATUM = "Tokyo Peil"

# Offset sign convention:
# "tp_minus_source" means station offsets are (TP - NLLW), positive if TP is above NLLW.
# With depth positive-down, conversion is:
#   depth_target = depth_source + offset
OFFSET_DEFINITION = "tp_minus_source"

OFFSET_METHOD = "idw"  # idw | nearest
IDW_POWER = 2.0
CHUNK_SIZE = 500000
SAVE_POINT_OFFSET = False  # save per-point offset arrays in point mode (can be large)
TSTATION_OUT_PATH = "/Users/kpark/Documents/DEM/M7000/M7005/tstation.out"

# Optional difference-map plotting (after - before depth, meters; positive means deeper)
PLOT_DIFF = True
GR3_PATH = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/01.gr3"
DIFF_PLOT_OUT = "/Users/kpark/Downloads/m7000_vdatum_depth_diff.png"
PLOT_SHOW = True
PLOT_KIND = "scatter"  # auto | contour | scatter
CONTOUR_LEVELS = 40
DIFF_CBAR_LIMS = None  # e.g., (-1.2, -0.4)
PLOT_MAX_POINTS = 200000
PLOT_ALPHA = 0.7
PLOT_SIZE = 2.0

# Tide-station offsets (approx. TP - NLLW, meters) and positions in decimal degrees.
STATION_OFFSETS = [
    {"pref": "Fukushima", "name": "Onahama", "offset_m": 0.87, "lat": 36.933333, "lon": 140.900000},
    {"pref": "Fukushima", "name": "Soma", "offset_m": 0.85, "lat": 37.833333, "lon": 140.950000},
    {"pref": "Miyagi", "name": "Ayukawa", "offset_m": 0.80, "lat": 38.300000, "lon": 141.500000},
    {"pref": "Miyagi", "name": "Sendai/Shiogama", "offset_m": 0.94, "lat": 38.316667, "lon": 141.033333},
    {"pref": "Miyagi", "name": "Ishinomaki", "offset_m": 0.82, "lat": 38.416667, "lon": 141.300000},
    {"pref": "Miyagi", "name": "Kesennuma", "offset_m": 0.88, "lat": 38.900000, "lon": 141.583333},
    {"pref": "Iwate", "name": "Ofunato", "offset_m": 0.88, "lat": 39.016667, "lon": 141.750000},
    {"pref": "Iwate", "name": "Kamaishi", "offset_m": 0.86, "lat": 39.266667, "lon": 141.883333},
    {"pref": "Iwate", "name": "Miyako", "offset_m": 0.84, "lat": 39.633333, "lon": 141.966667},
    {"pref": "Iwate", "name": "Kuji", "offset_m": 0.81, "lat": 40.200000, "lon": 141.783333},
    {"pref": "Aomori", "name": "Hachinohe", "offset_m": 0.75, "lat": 40.533333, "lon": 141.533333},
    {"pref": "Aomori", "name": "Ominato", "offset_m": 0.62, "lat": 41.250000, "lon": 141.150000},
]


def _parse_compact_dms(token):
    """
    Parse compact DMS string/number like:
      403200.0000   -> 40d 32m 00s
      1413200.0000  -> 141d 32m 00s
    Returns decimal degrees.
    """
    v = float(token)
    sign = -1.0 if v < 0 else 1.0
    v = abs(v)
    deg = int(v // 10000)
    rem = v - deg * 10000
    minute = int(rem // 100)
    sec = rem - minute * 100
    return sign * (deg + minute / 60.0 + sec / 3600.0)


def _read_tstation_out(path):
    """
    Read PatchJGD tstation.out.

    Expected numeric columns:
      lat_dms lon_dms before_offset_m corrected_offset_m
    The 4th numeric column is used as corrected offset.
    """
    stations = []
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            head = s.split("#", 1)[0].strip()
            if not head:
                continue
            parts = head.split()
            if len(parts) < 4:
                continue
            try:
                lat = _parse_compact_dms(parts[0])
                lon = _parse_compact_dms(parts[1])
                before_m = float(parts[2])
                after_m = float(parts[3])
            except Exception:
                continue
            if after_m <= -9999.0:
                # PatchJGD no-correction flag
                continue
            comment = ""
            if "#" in s:
                comment = s.split("#", 1)[1].strip()
            stations.append(
                {
                    "pref": "",
                    "name": comment or f"station_{len(stations)+1}",
                    "offset_m": after_m,
                    "offset_before_m": before_m,
                    "lat": lat,
                    "lon": lon,
                }
            )
    if not stations:
        raise ValueError(f"No valid station rows found in {p}")
    return stations


def _haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
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


def _station_arrays(stations):
    names = np.asarray([s["name"] for s in stations], dtype="U64")
    pref = np.asarray([s["pref"] for s in stations], dtype="U64")
    lat = np.asarray([s["lat"] for s in stations], dtype=np.float64)
    lon = np.asarray([s["lon"] for s in stations], dtype=np.float64)
    offset = np.asarray([s["offset_m"] for s in stations], dtype=np.float64)
    offset_before = np.asarray(
        [s.get("offset_before_m", np.nan) for s in stations], dtype=np.float64
    )
    return names, pref, lat, lon, offset, offset_before


def _interp_offsets(lat, lon, st_lat, st_lon, st_off, method, idw_power, chunk_size):
    """
    Interpolate offset field at M7000 points.

    Returns:
      offsets_m, nearest_station_dist_km
    """
    n = lat.size
    offsets = np.empty(n, dtype=np.float32)
    nearest_km = np.empty(n, dtype=np.float32)
    eps_m = 1e-6
    chunk = max(1, int(chunk_size))

    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        latc = lat[i0:i1][:, None]
        lonc = lon[i0:i1][:, None]

        d_m = _haversine_m(latc, lonc, st_lat[None, :], st_lon[None, :])
        nearest_idx = np.argmin(d_m, axis=1)
        nearest_km[i0:i1] = (d_m[np.arange(i1 - i0), nearest_idx] / 1000.0).astype(np.float32)

        if method == "nearest":
            offsets[i0:i1] = st_off[nearest_idx].astype(np.float32)
            continue

        # IDW with exact-station handling
        exact_mask = d_m <= eps_m
        if np.any(exact_mask):
            row_has_exact = np.any(exact_mask, axis=1)
        else:
            row_has_exact = np.zeros(i1 - i0, dtype=bool)

        out = np.empty(i1 - i0, dtype=np.float64)
        if np.any(row_has_exact):
            # If multiple exact hits (unlikely), use the first.
            exact_idx = np.argmax(exact_mask[row_has_exact], axis=1)
            out[row_has_exact] = st_off[exact_idx]

        if np.any(~row_has_exact):
            d_sel = d_m[~row_has_exact]
            w = 1.0 / np.power(d_sel, idw_power)
            out[~row_has_exact] = (w * st_off[None, :]).sum(axis=1) / w.sum(axis=1)

        offsets[i0:i1] = out.astype(np.float32)

    return offsets, nearest_km


def _apply_depth_conversion(depth_src, offsets_m, offset_definition):
    """
    Convert source depth to target depth using offset definition.

    depth arrays are assumed positive-down.
    """
    if offset_definition == "tp_minus_source":
        # depth_TP = depth_source + (TP - source)
        return (depth_src + offsets_m).astype(np.float32)
    if offset_definition == "source_minus_tp":
        # depth_TP = depth_source - (source - TP)
        return (depth_src - offsets_m).astype(np.float32)
    raise ValueError(f"Unsupported offset definition: {offset_definition}")


def _plot_diff_map(
    lat,
    lon,
    diff_m,
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
    station_lon=None,
    station_lat=None,
):
    """
    Plot depth difference map with:
    - jet colormap
    - x/y limits from GR3 domain bounds
    - colorbar scaling based on points inside grid domain only
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip diff plot.")
        return
    try:
        from pylib import read
    except ImportError:
        print("pylib not available; skip diff plot.")
        return

    gd = read(gr3_path)

    if diff_m.ndim == 2 and lat.ndim == 1 and lon.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
        lonp = lon2.ravel()
        latp = lat2.ravel()
        diffp = diff_m.ravel()
        grid_mode = True
    else:
        lonp = lon
        latp = lat
        diffp = diff_m
        grid_mode = False

    mask = np.isfinite(diffp)
    lonp = lonp[mask]
    latp = latp[mask]
    diffp = diffp[mask]

    xlim = ylim = None
    inside_mask = np.ones(diffp.shape, dtype=bool)
    if hasattr(gd, "x") and hasattr(gd, "y"):
        try:
            xlim = (float(np.nanmin(gd.x)), float(np.nanmax(gd.x)))
            ylim = (float(np.nanmin(gd.y)), float(np.nanmax(gd.y)))
        except Exception:
            xlim = ylim = None

    # Exact domain mask for color scaling when pylib provides inside_grid.
    try:
        pts = np.c_[lonp, latp]
        inside_raw = gd.inside_grid(pts)
        inside_mask = np.asarray(inside_raw).astype(bool).ravel()
        if inside_mask.size != diffp.size:
            inside_mask = np.ones(diffp.shape, dtype=bool)
    except Exception as exc:
        print(f"inside_grid mask unavailable, fallback to all finite points for color scaling: {exc}")
        inside_mask = np.ones(diffp.shape, dtype=bool)

    if lonp.size > max_points:
        step = max(1, lonp.size // max_points)
        lonp = lonp[::step]
        latp = latp[::step]
        diffp = diffp[::step]
        inside_mask = inside_mask[::step]
        print(f"Plot decimated to ~{lonp.size} points (step={step}).")

    # Colorbar range from inside-grid points only unless explicitly provided.
    if cbar_lims is None and np.any(inside_mask):
        vmin = float(np.nanmin(diffp[inside_mask]))
        vmax = float(np.nanmax(diffp[inside_mask]))
        cbar_lims_use = (vmin, vmax)
        print(
            "Diff color scale from in-grid points only (after-before, m):",
            f"vmin={vmin:.4f}, vmax={vmax:.4f}",
        )
    else:
        cbar_lims_use = cbar_lims

    fig, ax = plt.subplots(figsize=(8, 7))
    gd.plot_bnd()
    ax = plt.gca()

    use_contour = plot_kind in ("auto", "contour")
    artist = None
    vmin = vmax = None
    levels = _levels_from_lims(cbar_lims_use, contour_levels)
    if cbar_lims_use is not None:
        vmin, vmax = cbar_lims_use

    if use_contour and grid_mode:
        artist = ax.contourf(lon, lat, diff_m, levels=levels, cmap="jet")
    elif use_contour and lonp.size >= 3:
        try:
            artist = ax.tricontourf(lonp, latp, diffp, levels=levels, cmap="jet")
        except Exception as exc:
            print("tricontourf failed; fallback to scatter:", exc)
            artist = ax.scatter(
                lonp, latp, c=diffp, s=size, alpha=alpha, cmap="jet",
                vmin=vmin, vmax=vmax,
            )
    else:
        artist = ax.scatter(
            lonp, latp, c=diffp, s=size, alpha=alpha, cmap="jet",
            vmin=vmin, vmax=vmax,
        )

    cbar = fig.colorbar(artist, ax=ax)
    cbar.set_label("Depth Change (after - before, m)")
    if station_lon is not None and station_lat is not None:
        ax.plot(
            np.asarray(station_lon, dtype=float),
            np.asarray(station_lat, dtype=float),
            "m*",
            ms=10,
            label="Tide stations",
            zorder=10,
        )
        ax.legend(loc="best")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if plot_out:
        fig.savefig(plot_out, dpi=150)
        print("Saved diff plot to", plot_out)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Convert M7000 ASCII to NPZ and apply NLLW->Tokyo Peil vertical datum correction."
    )
    parser.add_argument("--input", default=INPUT_PATH, help="Input M7000 ASCII file")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output NPZ file")
    parser.add_argument("--mode", default=MODE, choices=["auto", "grid", "point"], help="Output mode")
    parser.add_argument("--grid-decimals", default=GRID_DECIMALS, type=int, help="Rounding decimals for grid detection")
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
        help="Flip input depth sign before vdatum conversion",
    )
    parser.add_argument(
        "--offset-method",
        default=OFFSET_METHOD,
        choices=["idw", "nearest"],
        help="Spatial interpolation method for TP-NLLW offsets",
    )
    parser.add_argument("--idw-power", default=IDW_POWER, type=float, help="IDW power (used when --offset-method idw)")
    parser.add_argument("--chunk-size", default=CHUNK_SIZE, type=int, help="Points per interpolation chunk")
    parser.add_argument(
        "--offset-definition",
        default=OFFSET_DEFINITION,
        choices=["tp_minus_source", "source_minus_tp"],
        help="Definition of station offsets supplied in STATION_OFFSETS",
    )
    parser.add_argument(
        "--save-point-offset",
        default=SAVE_POINT_OFFSET,
        action=argparse.BooleanOptionalAction,
        help="Save per-point offset and nearest station distance in point mode",
    )
    parser.add_argument(
        "--tstation-out",
        default=TSTATION_OUT_PATH,
        help="PatchJGD tstation.out file (4th column used as corrected TP-source offset). Empty to use built-in STATION_OFFSETS.",
    )
    parser.add_argument(
        "--plot-diff",
        default=PLOT_DIFF,
        action=argparse.BooleanOptionalAction,
        help="Plot 2D map of depth difference (after - before)",
    )
    parser.add_argument("--gr3", default=GR3_PATH, help="GR3 grid for boundary plot (used for diff map)")
    parser.add_argument("--diff-plot-out", default=DIFF_PLOT_OUT, help="Diff map output PNG")
    parser.add_argument(
        "--show",
        default=PLOT_SHOW,
        action=argparse.BooleanOptionalAction,
        help="Show diff plot interactively",
    )
    parser.add_argument(
        "--plot-kind",
        default=PLOT_KIND,
        choices=["auto", "contour", "scatter"],
        help="Diff plot style",
    )
    parser.add_argument("--contour-levels", default=CONTOUR_LEVELS, type=int, help="Number of contour levels for diff map")
    parser.add_argument(
        "--diff-cbar-lims",
        nargs=2,
        type=float,
        default=DIFF_CBAR_LIMS,
        metavar=("VMIN", "VMAX"),
        help="Colorbar limits for diff map (after - before depth)",
    )
    parser.add_argument("--plot-max-points", default=PLOT_MAX_POINTS, type=int, help="Max points to plot")
    parser.add_argument("--plot-alpha", default=PLOT_ALPHA, type=float, help="Diff scatter alpha")
    parser.add_argument("--plot-size", default=PLOT_SIZE, type=float, help="Diff scatter marker size")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    depth, lat, lon, rec_type, group, meta = _read_m7000(in_path)
    if args.flip_depth:
        depth = -depth

    print("Loaded", len(depth), "points from", in_path)
    print("Depth (source datum) min/max:", float(depth.min()), float(depth.max()))
    print("Lon range:", float(lon.min()), float(lon.max()), "Lat range:", float(lat.min()), float(lat.max()))

    if str(args.tstation_out).strip():
        stations = _read_tstation_out(args.tstation_out)
        print("Loaded corrected station offsets from tstation.out:", args.tstation_out)
    else:
        stations = STATION_OFFSETS
        print("Using built-in approximate station offsets (no --tstation-out)")

    st_names, st_pref, st_lat, st_lon, st_off, st_off_before = _station_arrays(stations)
    print("Using", len(st_off), "tide stations for offset interpolation")
    print(
        "Station TP-source offset (m) min/mean/max:",
        float(st_off.min()),
        float(st_off.mean()),
        float(st_off.max()),
    )
    if np.isfinite(st_off_before).any():
        delta_station = st_off.astype(np.float64) - st_off_before
        print(
            "Station offset update (corrected - original) (m) min/mean/max:",
            float(np.nanmin(delta_station)),
            float(np.nanmean(delta_station)),
            float(np.nanmax(delta_station)),
        )
        print("Sample station offsets (before -> corrected):")
        for i in range(min(6, len(st_off))):
            print(
                f"  {st_names[i]}: {float(st_off_before[i]):.3f} -> {float(st_off[i]):.3f} m"
            )

    offsets_m, nearest_km = _interp_offsets(
        lat=lat,
        lon=lon,
        st_lat=st_lat,
        st_lon=st_lon,
        st_off=st_off,
        method=args.offset_method,
        idw_power=args.idw_power,
        chunk_size=args.chunk_size,
    )
    depth_tp = _apply_depth_conversion(depth, offsets_m, args.offset_definition)
    depth_delta = (depth_tp - depth).astype(np.float32)  # after - before depth

    print(
        "Interpolated offset (m) min/mean/max:",
        float(np.nanmin(offsets_m)),
        float(np.nanmean(offsets_m)),
        float(np.nanmax(offsets_m)),
    )
    print(
        "Nearest station distance (km) min/median/max:",
        float(np.nanmin(nearest_km)),
        float(np.nanmedian(nearest_km)),
        float(np.nanmax(nearest_km)),
    )
    print("Depth (target datum) min/max:", float(np.nanmin(depth_tp)), float(np.nanmax(depth_tp)))
    print(
        "Depth change (after - before) (m) min/mean/max:",
        float(np.nanmin(depth_delta)),
        float(np.nanmean(depth_delta)),
        float(np.nanmax(depth_delta)),
    )

    payload = {
        "source": str(in_path),
        "source_vdatum": np.asarray(SOURCE_DATUM),
        "target_vdatum": np.asarray(TARGET_DATUM),
        "vdatum_offset_definition": np.asarray(args.offset_definition),
        "vdatum_method": np.asarray(args.offset_method),
        "vdatum_idw_power": np.asarray(float(args.idw_power)),
        "data_semantics": np.asarray("depth_positive_down"),
        "station_name": st_names,
        "station_pref": st_pref,
        "station_lat": st_lat.astype(np.float64),
        "station_lon": st_lon.astype(np.float64),
        "station_offset_m": st_off.astype(np.float32),
    }
    if np.isfinite(st_off_before).any():
        payload["station_offset_before_m"] = st_off_before.astype(np.float32)
        payload["station_offset_delta_m"] = (st_off.astype(np.float64) - st_off_before).astype(np.float32)
    if meta is not None:
        payload["meta"] = np.asarray(meta)
    if rec_type is not None:
        payload["rec_type"] = rec_type
    if group is not None:
        payload["group"] = group

    if args.mode in ("auto", "grid"):
        grid_depth = _as_grid(depth_tp, lat, lon, decimals=args.grid_decimals)
        grid_off = _as_grid(offsets_m, lat, lon, decimals=args.grid_decimals)
        grid_near = _as_grid(nearest_km, lat, lon, decimals=args.grid_decimals)
        if grid_depth is not None and grid_off is not None and grid_near is not None:
            uniq_lat, uniq_lon, depth_grid = grid_depth
            _, _, off_grid = grid_off
            _, _, near_grid = grid_near
            grid_delta = _as_grid(depth_delta, lat, lon, decimals=args.grid_decimals)
            _, _, diff_grid = grid_delta if grid_delta is not None else (None, None, None)
            payload.update(
                {
                    "lat": uniq_lat,
                    "lon": uniq_lon,
                    # keep key name 'elev' for compatibility with existing m2npz downstream usage
                    "elev": depth_grid.astype(np.float32),
                    "vdatum_offset_m": off_grid.astype(np.float32),
                    "nearest_station_km": near_grid.astype(np.float32),
                    "depth_delta_m": diff_grid.astype(np.float32) if diff_grid is not None else None,
                }
            )
            if payload.get("depth_delta_m") is None:
                payload.pop("depth_delta_m", None)
            _save_npz(out_path, payload, args.compress)
            print("Saved gridded TP-depth NPZ to", out_path)
            if args.plot_diff:
                _plot_diff_map(
                    payload["lat"],
                    payload["lon"],
                    payload["depth_delta_m"],
                    args.gr3,
                    args.diff_plot_out,
                    args.show,
                    args.plot_max_points,
                    args.plot_alpha,
                    args.plot_size,
                    args.plot_kind,
                    args.contour_levels,
                    args.diff_cbar_lims,
                    "M7000 Bathymetry Change (TP - NLLW depth; after - before, m)",
                    station_lon=st_lon,
                    station_lat=st_lat,
                )
            return 0
        if args.mode == "grid":
            raise SystemExit("Grid detection failed. Try --mode point.")

    payload.update(
        {
            "lat": lat.astype(np.float64),
            "lon": lon.astype(np.float64),
            "elev": depth_tp.astype(np.float32),  # compatibility key; contains TP-referenced depth
        }
    )
    payload["depth_delta_m"] = depth_delta.astype(np.float32)
    if args.save_point_offset:
        payload["vdatum_offset_m"] = offsets_m.astype(np.float32)
        payload["nearest_station_km"] = nearest_km.astype(np.float32)

    _save_npz(out_path, payload, args.compress)
    print("Saved point TP-depth NPZ to", out_path)
    if args.plot_diff:
        _plot_diff_map(
            payload["lat"],
            payload["lon"],
            payload["depth_delta_m"],
            args.gr3,
            args.diff_plot_out,
            args.show,
            args.plot_max_points,
            args.plot_alpha,
            args.plot_size,
            args.plot_kind,
            args.contour_levels,
            args.diff_cbar_lims,
            "M7000 Bathymetry Change (TP - NLLW depth; after - before, m)",
            station_lon=st_lon,
            station_lat=st_lat,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
