#!/usr/bin/env python3
from __future__ import annotations

"""
Compare TP-referenced M7000 bathymetry (NPZ) and ETOPO bathymetry on a SCHISM grid.

Goal:
- Diagnose potential merge-seam issues before blending M7000 (coastal) with ETOPO (offshore).
- Quantify differences near the M7000 coverage edge on the SCHISM grid.

Important (provisional mode):
- This script can compare ETOPO "as-is" (optionally converted from elevation to depth),
  but if ETOPO is not yet converted to Tokyo Peil (TP), the differences will include
  both bathymetric differences and vertical-datum differences.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from pylib import read
except Exception:  # pragma: no cover
    read = None


# ----------------------------
# Config (CLI overrides)
# ----------------------------
SCHISM_GR3 = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/03.gr3"
M7000_NPZ = "/Users/kpark/Downloads/M7005_TP.npz"
ETOPO_TIF = "/Users/kpark/Documents/DEM/ETOPO2022/15s_ice/15s_surface_elev_gtif/ETOPO_2022_v1_15s_N45E135_surface.tif"

OUT_DIR = "/Users/kpark/Downloads/m7000_etopo_compare"
OUT_PREFIX = "m7000_etopo_on_schism"

# Interpretation / conversion
M7000_KEY = "elev"  # m2npz/m2npz_vdatum compatibility key
M7000_IS_DEPTH = True  # expected True for m2npz_vdatum output
M7000_POINT_INTERP = "linear"  # nearest | idw | linear (used when M7000 NPZ is point-cloud)
M7000_POINT_K = 8               # neighbors for point-cloud IDW
M7000_POINT_MAX_DIST_KM = None  # optional max sampling radius; outside => NaN
ETOPO_IS_ELEVATION = True  # ETOPO "surface_elev" is elevation (+up); convert to depth
ETOPO_DEPTH_SIGN_FLIP = True  # if ETOPO_IS_ELEVATION, depth = -elevation
ETOPO_ADD_TP_BIAS_M = 0.0  # provisional constant bias adjustment to ETOPO depth to approximate TP

# Sampling
M7000_INTERP = "bilinear"  # nearest | bilinear (grid NPZ only)
SEAM_CORRIDOR_KM = 20.0    # corridor radius around M7000 coverage edge
MAX_PLOT_POINTS = 500000
PLOT = False
SHOW = False
# Difference config
DIFF_CBAR_LIMS = (-50,50)  # e.g. (-10, 10)
PLOT_ETOPO_CHECK = False
PLOT_DEPTH_COMPARE = False
DEPTH_CBAR_LIMS = None  # e.g. (0, 3000), shared for M7000 and ETOPO depth panels
DIFF_CMAP = "RdBu_r"
DEPTH_CMAP = "viridis_r"
PLOT_ROUGHNESS_SLOPE = False
ROUGHNESS_CBAR_LIMS = None  # e.g. (0, 50) meters
SLOPE_CBAR_LIMS = None      # e.g. (0, 0.05) m/m
ROUGHNESS_CMAP = "magma"
SLOPE_CMAP = "plasma"
# Density config
PLOT_M7000_DENSITY = False
M7000_DENSITY_K = 4                # kth-neighbor metric for density/trust
M7000_DENSITY_RADIUS_KM = 1.0      # local count radius on SCHISM nodes
M7000_ZONE_DK_MAX_KM = 0.4         # dk <= this => M7000-trusted zone
ETOPO_ZONE_DK_MIN_KM = 1        # dk >= this => ETOPO-trusted zone
DENSITY_DIST_CBAR_LIMS = None      # shared for d1/dk maps, km
DENSITY_COUNT_CBAR_LIMS = None     # local point-count map
DENSITY_DIST_CMAP = "jet"
DENSITY_COUNT_CMAP = "jet"


def _to_float_array(x):
    return np.asarray(x, dtype=np.float64).ravel()


def _load_schism_nodes(gr3_path):
    if read is None:
        raise SystemExit("pylib is required to read SCHISM grid (.gr3)")
    gd = read(gr3_path)
    x = _to_float_array(gd.x)
    y = _to_float_array(gd.y)
    return gd, x, y


def _normalize_axis(axis):
    a = np.asarray(axis, dtype=np.float64)
    if a.ndim != 1 or a.size < 2:
        raise ValueError("Axis must be 1D with at least 2 points")
    ascending = bool(a[-1] > a[0])
    if not ascending:
        a = a[::-1]
    return a, ascending


def _sample_regular_grid_nearest(lat_axis, lon_axis, grid, qlat, qlon):
    lat_use, lat_asc = _normalize_axis(lat_axis)
    lon_use, lon_asc = _normalize_axis(lon_axis)
    g = np.asarray(grid)
    if not lat_asc:
        g = g[::-1, :]
    if not lon_asc:
        g = g[:, ::-1]

    ilat = np.searchsorted(lat_use, qlat)
    ilat = np.clip(ilat, 1, len(lat_use) - 1)
    ilat_l = ilat - 1
    ilat_r = ilat
    ilat_n = np.where(np.abs(qlat - lat_use[ilat_l]) <= np.abs(qlat - lat_use[ilat_r]), ilat_l, ilat_r)

    ilon = np.searchsorted(lon_use, qlon)
    ilon = np.clip(ilon, 1, len(lon_use) - 1)
    ilon_l = ilon - 1
    ilon_r = ilon
    ilon_n = np.where(np.abs(qlon - lon_use[ilon_l]) <= np.abs(qlon - lon_use[ilon_r]), ilon_l, ilon_r)

    inside = (
        (qlat >= lat_use[0]) & (qlat <= lat_use[-1]) &
        (qlon >= lon_use[0]) & (qlon <= lon_use[-1])
    )
    out = np.full(qlat.shape, np.nan, dtype=np.float32)
    out[inside] = g[ilat_n[inside], ilon_n[inside]].astype(np.float32)
    return out


def _sample_regular_grid_bilinear(lat_axis, lon_axis, grid, qlat, qlon):
    lat_use, lat_asc = _normalize_axis(lat_axis)
    lon_use, lon_asc = _normalize_axis(lon_axis)
    g = np.asarray(grid, dtype=np.float64)
    if not lat_asc:
        g = g[::-1, :]
    if not lon_asc:
        g = g[:, ::-1]

    inside = (
        (qlat >= lat_use[0]) & (qlat <= lat_use[-1]) &
        (qlon >= lon_use[0]) & (qlon <= lon_use[-1])
    )
    out = np.full(qlat.shape, np.nan, dtype=np.float32)
    if not np.any(inside):
        return out

    qlat_i = qlat[inside]
    qlon_i = qlon[inside]

    j = np.searchsorted(lat_use, qlat_i) - 1
    i = np.searchsorted(lon_use, qlon_i) - 1
    j = np.clip(j, 0, len(lat_use) - 2)
    i = np.clip(i, 0, len(lon_use) - 2)

    lat0 = lat_use[j]
    lat1 = lat_use[j + 1]
    lon0 = lon_use[i]
    lon1 = lon_use[i + 1]

    wy = np.where(lat1 != lat0, (qlat_i - lat0) / (lat1 - lat0), 0.0)
    wx = np.where(lon1 != lon0, (qlon_i - lon0) / (lon1 - lon0), 0.0)

    v00 = g[j, i]
    v10 = g[j, i + 1]
    v01 = g[j + 1, i]
    v11 = g[j + 1, i + 1]

    # Bilinear with NaN-safe fallback to nearest when any corner is NaN
    vals = (
        (1 - wx) * (1 - wy) * v00 +
        wx * (1 - wy) * v10 +
        (1 - wx) * wy * v01 +
        wx * wy * v11
    )
    bad = ~(np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11))
    if np.any(bad):
        nn = _sample_regular_grid_nearest(lat_use, lon_use, g, qlat_i[bad], qlon_i[bad]).astype(np.float64)
        vals[bad] = nn

    out[inside] = vals.astype(np.float32)
    return out


def _sample_m7000_npz(
    npz_path,
    qlon,
    qlat,
    method,
    data_key,
    point_interp="nearest",
    point_k=8,
    point_max_dist_km=None,
):
    z = np.load(npz_path, allow_pickle=True)
    if "lat" not in z or "lon" not in z or data_key not in z:
        raise ValueError(f"NPZ missing required keys: lat/lon/{data_key}")
    lat = z["lat"]
    lon = z["lon"]
    data = z[data_key]

    if lat.ndim == 1 and lon.ndim == 1 and np.asarray(data).ndim == 2:
        if method == "nearest":
            v = _sample_regular_grid_nearest(lat, lon, data, qlat, qlon)
        elif method == "bilinear":
            v = _sample_regular_grid_bilinear(lat, lon, data, qlat, qlon)
        else:
            raise ValueError(f"Unsupported M7000 interp method: {method}")
        return v, z, "grid"

    if np.asarray(lat).ndim == 1 and np.asarray(lon).ndim == 1 and np.asarray(data).ndim == 1:
        if not (len(lat) == len(lon) == len(data)):
            raise ValueError("Point-cloud M7000 NPZ has inconsistent lat/lon/data lengths")
        v = _sample_points_to_nodes(
            px=lon,
            py=lat,
            pv=data,
            qx=qlon,
            qy=qlat,
            method=point_interp,
            k=point_k,
            max_dist_km=point_max_dist_km,
        )
        return v, z, "point"

    raise ValueError("Unsupported M7000 NPZ layout. Expected gridded or point-cloud lat/lon/elev.")


def _sample_etopo_tif(tif_path, qlon, qlat):
    try:
        import rasterio
    except Exception as exc:
        raise SystemExit(f"rasterio is required to sample ETOPO GeoTIFF: {exc}")

    vals = np.full(qlon.shape, np.nan, dtype=np.float32)
    with rasterio.open(tif_path) as ds:
        pts = list(zip(qlon.tolist(), qlat.tolist()))
        nodata = ds.nodata
        for i, arr in enumerate(ds.sample(pts)):
            if arr is None or len(arr) == 0:
                continue
            v = float(arr[0])
            if nodata is not None and np.isclose(v, nodata):
                continue
            if not np.isfinite(v):
                continue
            vals[i] = np.float32(v)
    return vals


def _plot_etopo_check(gd, tif_path, out_dir, prefix, show):
    """
    Plot ETOPO coverage (bounds, and raster subset if feasible) with SCHISM grid boundary.
    This is a diagnostic to catch CRS/coordinate-range mismatches.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skip ETOPO check plot.")
        return
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.plot import plotting_extent
    except Exception as exc:
        print(f"rasterio not available; skip ETOPO check plot: {exc}")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gx = _to_float_array(gd.x)
    gy = _to_float_array(gd.y)
    gb = (float(np.nanmin(gx)), float(np.nanmax(gx)), float(np.nanmin(gy)), float(np.nanmax(gy)))

    with rasterio.open(tif_path) as ds:
        print("ETOPO metadata:")
        print("  CRS   :", ds.crs)
        print("  Bounds:", ds.bounds)
        print("  Size  :", ds.width, "x", ds.height)
        print("  Nodata:", ds.nodata)
        print("  Dtype :", ds.dtypes[0] if ds.dtypes else "unknown")

        fig, ax = plt.subplots(figsize=(9, 8))

        # Try to draw a subset around the grid bbox (only if bbox overlaps raster bounds).
        rb = ds.bounds
        overlap = not (gb[1] < rb.left or gb[0] > rb.right or gb[3] < rb.bottom or gb[2] > rb.top)
        drew_raster = False
        if overlap:
            try:
                pad_x = 0.2 * (gb[1] - gb[0] if gb[1] > gb[0] else 1.0)
                pad_y = 0.2 * (gb[3] - gb[2] if gb[3] > gb[2] else 1.0)
                x0 = max(rb.left, gb[0] - pad_x)
                x1 = min(rb.right, gb[1] + pad_x)
                y0 = max(rb.bottom, gb[2] - pad_y)
                y1 = min(rb.top, gb[3] + pad_y)
                win = from_bounds(x0, y0, x1, y1, ds.transform)
                arr = ds.read(1, window=win, masked=True)
                ext = plotting_extent(arr, ds.transform, window=win)
                # Robust scaling for visualization
                data = np.asarray(arr.filled(np.nan), dtype=float)
                finite = data[np.isfinite(data)]
                vmin = vmax = None
                if finite.size:
                    vmin = float(np.nanpercentile(finite, 2))
                    vmax = float(np.nanpercentile(finite, 98))
                im = ax.imshow(
                    arr,
                    extent=ext,
                    origin="upper",
                    cmap="terrain",
                    vmin=vmin,
                    vmax=vmax,
                    alpha=0.85,
                )
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("ETOPO value (raw tif units)")
                drew_raster = True
            except Exception as exc:
                print(f"Failed to render ETOPO subset; plotting bounds only: {exc}")

        # Always draw ETOPO bounds rectangle
        ax.plot(
            [rb.left, rb.right, rb.right, rb.left, rb.left],
            [rb.bottom, rb.bottom, rb.top, rb.top, rb.bottom],
            "m-",
            lw=1.5,
            label="ETOPO bounds",
        )

        try:
            gd.plot_bnd(ax=ax)
        except Exception:
            gd.plot_bnd()
            ax = plt.gca()

        # Mark grid bbox too (helps diagnose projected-grid mismatch quickly)
        ax.plot(
            [gb[0], gb[1], gb[1], gb[0], gb[0]],
            [gb[2], gb[2], gb[3], gb[3], gb[2]],
            "k--",
            lw=1.0,
            label="SCHISM grid bbox",
        )
        ax.set_xlabel("X / Longitude")
        ax.set_ylabel("Y / Latitude")
        ax.set_title("ETOPO coverage check with SCHISM grid boundary")
        if drew_raster:
            # Focus on local region if subset drawn
            ax.set_xlim(min(gb[0], rb.left), max(gb[1], rb.right) if (rb.right-rb.left) < 10*(gb[1]-gb[0]) else gb[1] + 0.5*(gb[1]-gb[0]))
            ax.set_ylim(gb[2] - 0.5*(gb[3]-gb[2]), gb[3] + 0.5*(gb[3]-gb[2]))
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        out_png = out_dir / f"{prefix}_etopo_check.png"
        fig.savefig(out_png, dpi=150)
        print("Saved", out_png)
        if show:
            plt.show()
        plt.close(fig)


def _build_node_adjacency(gd, n_nodes):
    """
    Build SCHISM node adjacency from element connectivity.
    Expected pylib attrs: gd.elnode, gd.i34 (common).
    """
    adj = [set() for _ in range(n_nodes)]
    if not hasattr(gd, "elnode") or not hasattr(gd, "i34"):
        return adj

    elnode = np.asarray(gd.elnode)
    i34 = np.asarray(gd.i34).astype(int).ravel()
    ne = min(elnode.shape[0], i34.size)
    for e in range(ne):
        k = int(i34[e])
        if k <= 0:
            continue
        nodes = np.asarray(elnode[e, :k], dtype=int).ravel()
        nodes = nodes[nodes >= 0]  # pylib usually zero-based, -1 padded
        if nodes.size < 2:
            continue
        for a in range(nodes.size):
            ia = int(nodes[a])
            for b in range(a + 1, nodes.size):
                ib = int(nodes[b])
                if ia == ib:
                    continue
                adj[ia].add(ib)
                adj[ib].add(ia)
    return adj


def _m7000_edge_nodes(valid_mask, adjacency):
    edge = np.zeros(valid_mask.shape, dtype=bool)
    valid_idx = np.where(valid_mask)[0]
    for i in valid_idx:
        nbrs = adjacency[i]
        if not nbrs:
            continue
        for j in nbrs:
            if not valid_mask[j]:
                edge[i] = True
                break
    return edge


def _xy_to_local_km(x, y):
    """Approximate lon/lat to local km coordinates for distance queries."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lat0 = np.nanmean(y)
    xx = (x - np.nanmean(x)) * (111.32 * np.cos(np.deg2rad(lat0)))
    yy = (y - np.nanmean(y)) * 110.57
    return xx, yy


def _lonlat_to_local_km(x, y, lon0, lat0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xx = (x - lon0) * (111.32 * np.cos(np.deg2rad(lat0)))
    yy = (y - lat0) * 110.57
    return xx, yy


def _sample_points_to_nodes(px, py, pv, qx, qy, method="nearest", k=8, max_dist_km=None):
    """
    Sample point-cloud data (lon/lat/value) onto query nodes (lon/lat).
    Uses scipy for point-cloud interpolation; falls back to brute-force nearest only.
    """
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    pv = np.asarray(pv, dtype=np.float64).ravel()
    qx = np.asarray(qx, dtype=np.float64).ravel()
    qy = np.asarray(qy, dtype=np.float64).ravel()

    good = np.isfinite(px) & np.isfinite(py) & np.isfinite(pv)
    px, py, pv = px[good], py[good], pv[good]
    out = np.full(qx.shape, np.nan, dtype=np.float32)
    if px.size == 0:
        return out

    lon0 = float(np.nanmean(np.r_[px, qx]))
    lat0 = float(np.nanmean(np.r_[py, qy]))
    sx, sy = _lonlat_to_local_km(px, py, lon0, lat0)
    tx, ty = _lonlat_to_local_km(qx, qy, lon0, lat0)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.c_[sx, sy])
        if method == "nearest":
            dist, idx = tree.query(np.c_[tx, ty], k=1)
            dist = np.asarray(dist, dtype=np.float64)
            idx = np.asarray(idx, dtype=int)
            if max_dist_km is None:
                out[:] = pv[idx].astype(np.float32)
            else:
                keep = dist <= float(max_dist_km)
                out[keep] = pv[idx[keep]].astype(np.float32)
            return out

        if method == "idw":
            kk = max(1, int(k))
            dist, idx = tree.query(np.c_[tx, ty], k=kk)
            dist = np.asarray(dist, dtype=np.float64)
            idx = np.asarray(idx, dtype=int)
            if kk == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            vals = pv[idx]
            eps = 1e-12
            exact = dist <= eps
            row_exact = np.any(exact, axis=1)
            if np.any(row_exact):
                ex_col = np.argmax(exact[row_exact], axis=1)
                out[row_exact] = vals[row_exact, ex_col].astype(np.float32)

            rem_rows = np.where(~row_exact)[0]
            if rem_rows.size:
                d = dist[rem_rows].copy()
                v = vals[rem_rows]
                if max_dist_km is not None:
                    m = d <= float(max_dist_km)
                    d[~m] = np.nan
                    row_has = np.any(m, axis=1)
                    if np.any(row_has):
                        d2 = d[row_has]
                        v2 = v[row_has]
                        w = 1.0 / np.power(d2, 2.0)
                        out[rem_rows[row_has]] = (np.nansum(w * v2, axis=1) / np.nansum(w, axis=1)).astype(np.float32)
                else:
                    w = 1.0 / np.power(d, 2.0)
                    out[rem_rows] = (np.sum(w * v, axis=1) / np.sum(w, axis=1)).astype(np.float32)
            return out

        if method == "linear":
            try:
                from scipy.interpolate import LinearNDInterpolator  # type: ignore
            except Exception as iexc:
                raise RuntimeError(
                    f"Point-cloud linear interpolation requires scipy.interpolate.LinearNDInterpolator: {iexc}"
                ) from iexc
            interp = LinearNDInterpolator(np.c_[sx, sy], pv, fill_value=np.nan)
            vals = np.asarray(interp(np.c_[tx, ty]), dtype=np.float64).ravel()
            if max_dist_km is not None:
                dist1, _ = tree.query(np.c_[tx, ty], k=1)
                dist1 = np.asarray(dist1, dtype=np.float64)
                vals[dist1 > float(max_dist_km)] = np.nan
            out[np.isfinite(vals)] = vals[np.isfinite(vals)].astype(np.float32)
            return out

        raise ValueError(f"Unsupported point interpolation method: {method}")

    except Exception as exc:
        if method != "nearest":
            raise RuntimeError(
                f"Point-cloud '{method}' interpolation requires scipy (and may need cKDTree/LinearNDInterpolator): {exc}"
            ) from exc
        if px.size * qx.size > 2e8:
            raise RuntimeError(
                "Point-cloud nearest fallback without scipy is too expensive. Install scipy."
            ) from exc
        chunk = 5000
        for i0 in range(0, qx.size, chunk):
            i1 = min(qx.size, i0 + chunk)
            dx = tx[i0:i1, None] - sx[None, :]
            dy = ty[i0:i1, None] - sy[None, :]
            d = np.hypot(dx, dy)
            idx = np.argmin(d, axis=1)
            dmin = d[np.arange(i1 - i0), idx]
            if max_dist_km is None:
                out[i0:i1] = pv[idx].astype(np.float32)
            else:
                keep = dmin <= float(max_dist_km)
                if np.any(keep):
                    out[i0:i1][keep] = pv[idx[keep]].astype(np.float32)
        return out


def _distance_to_edge_km(x, y, edge_mask):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if not np.any(edge_mask):
        return out
    xe, ye = _xy_to_local_km(x[edge_mask], y[edge_mask])
    xq, yq = _xy_to_local_km(x, y)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.c_[xe, ye])
        dist, _ = tree.query(np.c_[xq, yq], k=1)
        out[:] = dist.astype(np.float32)
        return out
    except Exception:
        # Fallback chunked brute-force (slower but dependency-free)
        chunk = 20000
        for i0 in range(0, x.size, chunk):
            i1 = min(x.size, i0 + chunk)
            dx = xq[i0:i1, None] - xe[None, :]
            dy = yq[i0:i1, None] - ye[None, :]
            d = np.hypot(dx, dy)
            out[i0:i1] = np.nanmin(d, axis=1).astype(np.float32)
        return out


def _compute_m7000_point_density_metrics(px, py, qx, qy, k=8, radius_km=1.0):
    """
    Density/trust diagnostics from raw M7000 point-cloud support around SCHISM nodes.
    Returns:
      d1_km   : nearest-point distance [km]
      dk_km   : k-th nearest-point distance [km]
      count_r : count of points within radius_km
    """
    px = np.asarray(px, dtype=np.float64).ravel()
    py = np.asarray(py, dtype=np.float64).ravel()
    qx = np.asarray(qx, dtype=np.float64).ravel()
    qy = np.asarray(qy, dtype=np.float64).ravel()
    n = qx.size

    d1 = np.full(n, np.nan, dtype=np.float32)
    dk = np.full(n, np.nan, dtype=np.float32)
    cnt = np.zeros(n, dtype=np.int32)

    good = np.isfinite(px) & np.isfinite(py)
    px = px[good]
    py = py[good]
    if px.size == 0:
        return d1, dk, cnt

    kk = max(1, int(k))
    rr = max(0.0, float(radius_km))
    lon0 = float(np.nanmean(np.r_[px, qx]))
    lat0 = float(np.nanmean(np.r_[py, qy]))
    sx, sy = _lonlat_to_local_km(px, py, lon0, lat0)
    tx, ty = _lonlat_to_local_km(qx, qy, lon0, lat0)
    qp = np.c_[tx, ty]

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(np.c_[sx, sy])
        dist, _ = tree.query(qp, k=kk)
        dist = np.asarray(dist, dtype=np.float64)
        if kk == 1:
            dist = dist[:, None]
        d1[:] = dist[:, 0].astype(np.float32)
        dk[:] = dist[:, -1].astype(np.float32)

        if rr > 0.0:
            chunk = 10000
            for i0 in range(0, n, chunk):
                i1 = min(n, i0 + chunk)
                hits = tree.query_ball_point(qp[i0:i1], r=rr)
                cnt[i0:i1] = np.asarray([len(h) for h in hits], dtype=np.int32)
        return d1, dk, cnt
    except Exception as exc:
        if px.size * n > 2e8:
            print(f"[WARN] Skip point-density metrics without scipy.cKDTree (problem too large): {exc}")
            return d1, dk, cnt
        chunk = 5000
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            dx = tx[i0:i1, None] - sx[None, :]
            dy = ty[i0:i1, None] - sy[None, :]
            d = np.hypot(dx, dy)
            d_sorted = np.sort(d, axis=1)
            d1[i0:i1] = d_sorted[:, 0].astype(np.float32)
            dk_col = min(kk - 1, d_sorted.shape[1] - 1)
            dk[i0:i1] = d_sorted[:, dk_col].astype(np.float32)
            if rr > 0.0:
                cnt[i0:i1] = np.sum(d <= rr, axis=1).astype(np.int32)
        return d1, dk, cnt


def _classify_density_zones(dk_km, m7000_dk_max_km, etopo_dk_min_km):
    """
    Zone code (int8):
      0 = unknown (non-finite)
      1 = M7000 zone
      2 = blend zone
      3 = ETOPO zone
    """
    dk = np.asarray(dk_km, dtype=np.float64)
    z = np.zeros(dk.shape, dtype=np.int8)
    finite = np.isfinite(dk)
    if not np.any(finite):
        return z
    a = float(m7000_dk_max_km)
    b = float(etopo_dk_min_km)
    if a > b:
        a, b = b, a
    z[finite & (dk <= a)] = 1
    z[finite & (dk >= b)] = 3
    z[finite & (dk > a) & (dk < b)] = 2
    return z


def _compute_local_roughness_slope(x, y, z, valid_mask, adjacency):
    """
    Compute simple local diagnostics on SCHISM nodes using immediate neighbors:
    - roughness_m: median(|dz|) over valid neighbors [m]
    - slope_proxy: median(|dz| / ds) over valid neighbors [m/m]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    n = z.size
    rough = np.full(n, np.nan, dtype=np.float32)
    slope = np.full(n, np.nan, dtype=np.float32)

    xk, yk = _xy_to_local_km(x, y)
    xm = xk * 1000.0
    ym = yk * 1000.0

    for i in range(n):
        if not valid_mask[i] or not np.isfinite(z[i]):
            continue
        nbrs = adjacency[i]
        if not nbrs:
            continue
        nbr = np.fromiter(nbrs, dtype=int, count=len(nbrs))
        if nbr.size == 0:
            continue
        keep = valid_mask[nbr] & np.isfinite(z[nbr])
        if not np.any(keep):
            continue
        nbr = nbr[keep]
        dz = np.abs(z[nbr] - z[i])
        ds = np.hypot(xm[nbr] - xm[i], ym[nbr] - ym[i])
        ok = np.isfinite(dz) & np.isfinite(ds) & (ds > 0.0)
        if not np.any(ok):
            continue
        dz = dz[ok]
        ds = ds[ok]
        rough[i] = np.float32(np.median(dz))
        slope[i] = np.float32(np.median(dz / ds))

    return rough, slope


def _stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "p05": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "std": float(np.std(a)),
    }


def _print_stats(label, arr):
    s = _stats(arr)
    if s is None:
        print(f"{label}: no valid values")
        return
    print(
        f"{label}: n={s['count']} min={s['min']:.3f} p05={s['p05']:.3f} "
        f"median={s['median']:.3f} mean={s['mean']:.3f} p95={s['p95']:.3f} "
        f"max={s['max']:.3f} std={s['std']:.3f}"
    )


def _write_stats_csv(out_csv, rows):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = ["label", "count", "min", "p05", "median", "mean", "p95", "max", "std"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_maps(gd, x, y, diff, overlap_mask, seam_mask, edge_mask, out_dir, prefix, show, max_plot_points, cbar_lims, diff_cmap):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skip plots.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decimate for plotting
    idx = np.where(overlap_mask & np.isfinite(diff))[0]
    if idx.size == 0:
        print("No overlap diff points to plot.")
        return
    if idx.size > max_plot_points:
        step = max(1, idx.size // max_plot_points)
        idx = idx[::step]
        print(f"Plot decimated to ~{idx.size} overlap points (step={step}).")

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    try:
        gd.plot_bnd(ax=ax)
    except Exception:
        gd.plot_bnd()
        ax = plt.gca()
    sc = ax.scatter(
        x[idx], y[idx], c=diff[idx], s=3, alpha=0.7, cmap=diff_cmap,
        vmin=(cbar_lims[0] if cbar_lims else None),
        vmax=(cbar_lims[1] if cbar_lims else None),
    )
    edge_idx = np.where(edge_mask)[0]
    if edge_idx.size:
        ax.plot(x[edge_idx], y[edge_idx], "k.", ms=1.5, alpha=0.5, label="M7000 coverage edge")
    seam_idx = np.where(seam_mask & overlap_mask & np.isfinite(diff))[0]
    if seam_idx.size:
        nmark = seam_idx[:: max(1, seam_idx.size // 20000)]
        ax.plot(x[nmark], y[nmark], "g.", ms=1.0, alpha=0.4, label="Seam corridor nodes")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("M7000(TP depth) - ETOPO(depth, provisional TP adj) [m]")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("M7000 vs ETOPO difference on SCHISM nodes")
    try:
        ax.set_xlim(float(np.nanmin(gd.x)), float(np.nanmax(gd.x)))
        ax.set_ylim(float(np.nanmin(gd.y)), float(np.nanmax(gd.y)))
    except Exception:
        pass
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    out_png = out_dir / f"{prefix}_diff_map.png"
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _plot_depth_compare_maps(
    gd,
    x,
    y,
    m7000_depth,
    etopo_depth,
    valid_m7000,
    valid_etopo,
    out_dir,
    prefix,
    show,
    max_plot_points,
    cbar_lims,
    depth_cmap,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skip depth comparison plots.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_m = np.where(valid_m7000 & np.isfinite(m7000_depth))[0]
    idx_e = np.where(valid_etopo & np.isfinite(etopo_depth))[0]
    if idx_m.size == 0 and idx_e.size == 0:
        print("No valid M7000/ETOPO depth nodes to plot.")
        return

    if idx_m.size > max_plot_points:
        step = max(1, idx_m.size // max_plot_points)
        idx_m = idx_m[::step]
        print(f"M7000 depth plot decimated to ~{idx_m.size} points (step={step}).")
    if idx_e.size > max_plot_points:
        step = max(1, idx_e.size // max_plot_points)
        idx_e = idx_e[::step]
        print(f"ETOPO depth plot decimated to ~{idx_e.size} points (step={step}).")

    if cbar_lims is None:
        vals = np.r_[
            m7000_depth[valid_m7000 & np.isfinite(m7000_depth)],
            etopo_depth[valid_etopo & np.isfinite(etopo_depth)],
        ]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            vmin = float(np.nanpercentile(vals, 2))
            vmax = float(np.nanpercentile(vals, 98))
            if vmin == vmax:
                vmax = vmin + 1.0
            cbar_lims_use = (vmin, vmax)
        else:
            cbar_lims_use = None
    else:
        cbar_lims_use = cbar_lims

    vmin = cbar_lims_use[0] if cbar_lims_use else None
    vmax = cbar_lims_use[1] if cbar_lims_use else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True)
    last_artist = None
    for ax, idx, z, title in [
        (axes[0], idx_m, m7000_depth, "M7000 depth on SCHISM nodes (TP)"),
        (axes[1], idx_e, etopo_depth, "ETOPO depth on SCHISM nodes (provisional)"),
    ]:
        try:
            gd.plot_bnd(ax=ax)
        except Exception:
            plt.sca(ax)
            gd.plot_bnd()
            ax = plt.gca()
        if idx.size:
            last_artist = ax.scatter(
                x[idx], y[idx], c=z[idx], s=3, alpha=0.7, cmap=depth_cmap,
                vmin=vmin, vmax=vmax,
            )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        try:
            ax.set_xlim(float(np.nanmin(gd.x)), float(np.nanmax(gd.x)))
            ax.set_ylim(float(np.nanmin(gd.y)), float(np.nanmax(gd.y)))
        except Exception:
            pass
        ax.set_aspect("equal", adjustable="box")

    if last_artist is not None:
        cbar = fig.colorbar(last_artist, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03, shrink=0.96)
        cbar.set_label("Depth (m, positive-down)")
    if cbar_lims_use is not None:
        print(f"Depth compare color scale (shared): vmin={cbar_lims_use[0]:.3f}, vmax={cbar_lims_use[1]:.3f}")
    out_png = out_dir / f"{prefix}_depth_compare.png"
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)

    # Direct overlap scatter helps evaluate which dataset is biased/noisy.
    ov = valid_m7000 & valid_etopo & np.isfinite(m7000_depth) & np.isfinite(etopo_depth)
    idx_o = np.where(ov)[0]
    if idx_o.size == 0:
        return
    if idx_o.size > max_plot_points:
        step = max(1, idx_o.size // max_plot_points)
        idx_o = idx_o[::step]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(etopo_depth[idx_o], m7000_depth[idx_o], s=4, alpha=0.3)
    finite = np.r_[etopo_depth[idx_o], m7000_depth[idx_o]]
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo = float(np.nanpercentile(finite, 1))
        hi = float(np.nanpercentile(finite, 99))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("ETOPO depth (m)")
    ax.set_ylabel("M7000 depth (m, TP)")
    ax.set_title("Depth scatter on overlap nodes")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_png = out_dir / f"{prefix}_depth_scatter.png"
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _plot_pair_maps(
    gd,
    x,
    y,
    z1,
    z2,
    valid1,
    valid2,
    out_png,
    show,
    max_plot_points,
    cbar_lims,
    cmap,
    titles,
    cbar_label,
):
    """Generic side-by-side node maps with shared color scale."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skip pair plot.")
        return

    idx1 = np.where(valid1 & np.isfinite(z1))[0]
    idx2 = np.where(valid2 & np.isfinite(z2))[0]
    if idx1.size == 0 and idx2.size == 0:
        print(f"No valid values to plot for {out_png.name}")
        return
    if idx1.size > max_plot_points:
        idx1 = idx1[:: max(1, idx1.size // max_plot_points)]
    if idx2.size > max_plot_points:
        idx2 = idx2[:: max(1, idx2.size // max_plot_points)]

    if cbar_lims is None:
        vals = np.r_[z1[valid1 & np.isfinite(z1)], z2[valid2 & np.isfinite(z2)]]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            vmin = float(np.nanpercentile(vals, 2))
            vmax = float(np.nanpercentile(vals, 98))
            if vmin == vmax:
                vmax = vmin + 1e-6
            cbar_use = (vmin, vmax)
        else:
            cbar_use = None
    else:
        cbar_use = cbar_lims
    vmin = cbar_use[0] if cbar_use else None
    vmax = cbar_use[1] if cbar_use else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True)
    last = None
    for ax, idx, z, title in [
        (axes[0], idx1, z1, titles[0]),
        (axes[1], idx2, z2, titles[1]),
    ]:
        try:
            gd.plot_bnd(ax=ax)
        except Exception:
            plt.sca(ax)
            gd.plot_bnd()
            ax = plt.gca()
        if idx.size:
            last = ax.scatter(x[idx], y[idx], c=z[idx], s=3, alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        try:
            ax.set_xlim(float(np.nanmin(gd.x)), float(np.nanmax(gd.x)))
            ax.set_ylim(float(np.nanmin(gd.y)), float(np.nanmax(gd.y)))
        except Exception:
            pass
        ax.set_aspect("equal", adjustable="box")
    if last is not None:
        cbar = fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03, shrink=0.96)
        cbar.set_label(cbar_label)
    if cbar_use is not None:
        print(f"{out_png.stem} color scale: vmin={cbar_use[0]:.4g}, vmax={cbar_use[1]:.4g}")
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _plot_overlap_scatter(xv, yv, out_png, show, xlabel, ylabel, title, max_plot_points):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skip scatter plot.")
        return
    m = np.isfinite(xv) & np.isfinite(yv)
    idx = np.where(m)[0]
    if idx.size == 0:
        return
    if idx.size > max_plot_points:
        idx = idx[:: max(1, idx.size // max_plot_points)]
    x1 = xv[idx]
    y1 = yv[idx]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(x1, y1, s=4, alpha=0.3)
    finite = np.r_[x1, y1]
    finite = finite[np.isfinite(finite)]
    if finite.size:
        lo = float(np.nanpercentile(finite, 1))
        hi = float(np.nanpercentile(finite, 99))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _plot_single_node_map(
    gd,
    x,
    y,
    z,
    valid_mask,
    out_png,
    show,
    max_plot_points,
    cbar_lims,
    cmap,
    title,
    cbar_label,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"matplotlib not available; skip {Path(out_png).name}")
        return

    idx = np.where(valid_mask & np.isfinite(z))[0]
    if idx.size == 0:
        print(f"No valid values to plot for {Path(out_png).name}")
        return
    if idx.size > max_plot_points:
        idx = idx[:: max(1, idx.size // max_plot_points)]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    try:
        gd.plot_bnd(ax=ax)
    except Exception:
        gd.plot_bnd()
        ax = plt.gca()
    sc = ax.scatter(
        x[idx], y[idx], c=z[idx], s=3, alpha=0.7, cmap=cmap,
        vmin=(cbar_lims[0] if cbar_lims else None),
        vmax=(cbar_lims[1] if cbar_lims else None),
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    try:
        ax.set_xlim(float(np.nanmin(gd.x)), float(np.nanmax(gd.x)))
        ax.set_ylim(float(np.nanmin(gd.y)), float(np.nanmax(gd.y)))
    except Exception:
        pass
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _plot_m7000_density_diagnostics(
    gd,
    x,
    y,
    d1_km,
    dk_km,
    count_r,
    zone_code,
    k,
    radius_km,
    m7000_zone_dk_max_km,
    etopo_zone_dk_min_km,
    out_dir,
    prefix,
    show,
    max_plot_points,
    dist_cbar_lims,
    count_cbar_lims,
    dist_cmap,
    count_cmap,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except Exception:
        print("matplotlib not available; skip M7000 density diagnostics.")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finite_d1 = np.isfinite(d1_km)
    finite_dk = np.isfinite(dk_km)
    finite_cnt = np.isfinite(count_r)

    _plot_single_node_map(
        gd, x, y, d1_km, finite_d1,
        out_dir / f"{prefix}_m7000_density_d1_km.png",
        show, max_plot_points, dist_cbar_lims, dist_cmap,
        "M7000 support distance on SCHISM nodes (nearest point, d1)",
        "Distance to nearest M7000 point (km)",
    )
    _plot_single_node_map(
        gd, x, y, dk_km, finite_dk,
        out_dir / f"{prefix}_m7000_density_d{k}_km.png",
        show, max_plot_points, dist_cbar_lims, dist_cmap,
        f"M7000 support distance on SCHISM nodes (d{k})",
        f"Distance to {k}-th nearest M7000 point (km)",
    )
    _plot_single_node_map(
        gd, x, y, count_r.astype(np.float32), finite_cnt,
        out_dir / f"{prefix}_m7000_density_count_r{radius_km:g}km.png",
        show, max_plot_points, count_cbar_lims, count_cmap,
        f"M7000 local point count on SCHISM nodes (r={radius_km:g} km)",
        f"M7000 point count within {radius_km:g} km",
    )

    # Zone classification map
    idx = np.where(zone_code > 0)[0]
    if idx.size:
        if idx.size > max_plot_points:
            idx = idx[:: max(1, idx.size // max_plot_points)]
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        try:
            gd.plot_bnd(ax=ax)
        except Exception:
            gd.plot_bnd()
            ax = plt.gca()
        cmap_zone = ListedColormap(["#1f77b4", "#ffbf00", "#d62728"])  # M7000, Blend, ETOPO
        norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap_zone.N)
        sc = ax.scatter(x[idx], y[idx], c=zone_code[idx], s=3, alpha=0.8, cmap=cmap_zone, norm=norm)
        cbar = fig.colorbar(sc, ax=ax, ticks=[1, 2, 3])
        cbar.ax.set_yticklabels(["M7000", "Blend", "ETOPO"])
        cbar.set_label("Density-derived zone")
        ax.set_title(
            f"M7000 density-derived zones on SCHISM nodes (d{k} thresholds: "
            f"M7000<={m7000_zone_dk_max_km:g} km, ETOPO>={etopo_zone_dk_min_km:g} km)"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        try:
            ax.set_xlim(float(np.nanmin(gd.x)), float(np.nanmax(gd.x)))
            ax.set_ylim(float(np.nanmin(gd.y)), float(np.nanmax(gd.y)))
        except Exception:
            pass
        ax.set_aspect("equal", adjustable="box")
        out_png = out_dir / f"{prefix}_m7000_density_zone_map.png"
        fig.savefig(out_png, dpi=150)
        print("Saved", out_png)
        if show:
            plt.show()
        plt.close(fig)

    # Histograms for threshold selection
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    a = np.asarray(dk_km, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size:
        axes[0].hist(a, bins=80, color="#4c78a8", alpha=0.85)
        axes[0].axvline(float(m7000_zone_dk_max_km), color="g", ls="--", lw=1.5, label="M7000 zone threshold")
        axes[0].axvline(float(etopo_zone_dk_min_km), color="r", ls="--", lw=1.5, label="ETOPO zone threshold")
        axes[0].set_xlabel(f"d{k} distance (km)")
        axes[0].set_ylabel("Node count")
        axes[0].set_title(f"M7000 d{k} distribution on SCHISM nodes")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend(loc="best")
    b = np.asarray(count_r, dtype=np.float64)
    b = b[np.isfinite(b)]
    if b.size:
        axes[1].hist(b, bins=min(80, max(20, int(np.nanmax(b)) + 1)), color="#f58518", alpha=0.85)
        axes[1].set_xlabel(f"Point count within {radius_km:g} km")
        axes[1].set_ylabel("Node count")
        axes[1].set_title("Local M7000 point-count distribution")
        axes[1].grid(True, alpha=0.2)
    out_png = out_dir / f"{prefix}_m7000_density_hist.png"
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _plot_roughness_slope_diagnostics(
    gd,
    x,
    y,
    m7000_depth,
    etopo_depth,
    valid_m7000,
    valid_etopo,
    adjacency,
    out_dir,
    prefix,
    show,
    max_plot_points,
    roughness_cbar_lims,
    slope_cbar_lims,
    roughness_cmap,
    slope_cmap,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing local roughness/slope diagnostics on SCHISM nodes...")
    m7000_rough, m7000_slope = _compute_local_roughness_slope(x, y, m7000_depth, valid_m7000, adjacency)
    etopo_rough, etopo_slope = _compute_local_roughness_slope(x, y, etopo_depth, valid_etopo, adjacency)

    ov = valid_m7000 & valid_etopo
    _print_stats("Roughness diff (M7000-ETOPO) overlap", m7000_rough[ov] - etopo_rough[ov])
    _print_stats("Slope diff (M7000-ETOPO) overlap", m7000_slope[ov] - etopo_slope[ov])

    _plot_pair_maps(
        gd=gd,
        x=x,
        y=y,
        z1=m7000_rough,
        z2=etopo_rough,
        valid1=valid_m7000,
        valid2=valid_etopo,
        out_png=out_dir / f"{prefix}_roughness_compare.png",
        show=show,
        max_plot_points=max_plot_points,
        cbar_lims=roughness_cbar_lims,
        cmap=roughness_cmap,
        titles=("M7000 local roughness (median |dz|, m)", "ETOPO local roughness (median |dz|, m)"),
        cbar_label="Local roughness (m)",
    )
    _plot_pair_maps(
        gd=gd,
        x=x,
        y=y,
        z1=m7000_slope,
        z2=etopo_slope,
        valid1=valid_m7000,
        valid2=valid_etopo,
        out_png=out_dir / f"{prefix}_slope_compare.png",
        show=show,
        max_plot_points=max_plot_points,
        cbar_lims=slope_cbar_lims,
        cmap=slope_cmap,
        titles=("M7000 slope proxy (median |dz|/ds, m/m)", "ETOPO slope proxy (median |dz|/ds, m/m)"),
        cbar_label="Slope proxy (m/m)",
    )
    _plot_overlap_scatter(
        etopo_rough,
        m7000_rough,
        out_dir / f"{prefix}_roughness_scatter.png",
        show,
        xlabel="ETOPO roughness (m)",
        ylabel="M7000 roughness (m)",
        title="Local roughness scatter on overlap nodes",
        max_plot_points=max_plot_points,
    )
    _plot_overlap_scatter(
        etopo_slope,
        m7000_slope,
        out_dir / f"{prefix}_slope_scatter.png",
        show,
        xlabel="ETOPO slope proxy (m/m)",
        ylabel="M7000 slope proxy (m/m)",
        title="Local slope proxy scatter on overlap nodes",
        max_plot_points=max_plot_points,
    )

def main():
    ap = argparse.ArgumentParser(description="Compare M7000 and ETOPO bathymetry on a SCHISM grid.")
    ap.add_argument("--gr3", default=SCHISM_GR3, help="SCHISM grid (.gr3)")
    ap.add_argument("--m7000-npz", default=M7000_NPZ, help="M7000 NPZ (prefer TP-corrected gridded output)")
    ap.add_argument("--etopo-tif", default=ETOPO_TIF, help="ETOPO GeoTIFF")
    ap.add_argument("--out-dir", default=OUT_DIR, help="Output directory")
    ap.add_argument("--out-prefix", default=OUT_PREFIX, help="Output filename prefix")
    ap.add_argument("--m7000-key", default=M7000_KEY, help="M7000 data key in NPZ (default: elev)")
    ap.add_argument("--m7000-interp", default=M7000_INTERP, choices=["nearest", "bilinear"], help="M7000 regular-grid interpolation")
    ap.add_argument("--m7000-point-interp", default=M7000_POINT_INTERP, choices=["nearest", "idw", "linear"], help="Point-cloud M7000 interpolation")
    ap.add_argument("--m7000-point-k", default=M7000_POINT_K, type=int, help="Neighbors for point-cloud IDW")
    ap.add_argument("--m7000-point-max-dist-km", default=M7000_POINT_MAX_DIST_KM, type=float, help="Optional max distance for point-cloud sampling")
    ap.add_argument("--seam-corridor-km", default=SEAM_CORRIDOR_KM, type=float, help="Seam corridor radius from M7000 coverage edge")
    ap.add_argument("--etopo-is-elevation", default=ETOPO_IS_ELEVATION, action=argparse.BooleanOptionalAction, help="ETOPO tif values are elevation (+up)")
    ap.add_argument("--etopo-depth-sign-flip", default=ETOPO_DEPTH_SIGN_FLIP, action=argparse.BooleanOptionalAction, help="Convert ETOPO elevation to depth by negating")
    ap.add_argument("--etopo-add-tp-bias-m", default=ETOPO_ADD_TP_BIAS_M, type=float, help="Provisional constant bias added to ETOPO depth")
    ap.add_argument("--plot", default=PLOT, action=argparse.BooleanOptionalAction, help="Write diagnostic plots")
    ap.add_argument("--plot-etopo-check", default=PLOT_ETOPO_CHECK, action=argparse.BooleanOptionalAction, help="Write ETOPO coverage check plot with SCHISM boundary")
    ap.add_argument("--plot-depth-compare", default=PLOT_DEPTH_COMPARE, action=argparse.BooleanOptionalAction, help="Write side-by-side M7000 and ETOPO depth maps on SCHISM nodes")
    ap.add_argument("--plot-roughness-slope", default=PLOT_ROUGHNESS_SLOPE, action=argparse.BooleanOptionalAction, help="Write local roughness/slope proxy diagnostics on SCHISM nodes")
    ap.add_argument("--plot-m7000-density", default=PLOT_M7000_DENSITY, action=argparse.BooleanOptionalAction, help="Write M7000 point-density diagnostics and zone map (point-cloud M7000 only)")
    ap.add_argument("--show", default=SHOW, action=argparse.BooleanOptionalAction, help="Show plots interactively")
    ap.add_argument("--max-plot-points", default=MAX_PLOT_POINTS, type=int, help="Max nodes plotted")
    ap.add_argument("--m7000-density-k", default=M7000_DENSITY_K, type=int, help="k for M7000 point-density d_k metric on SCHISM nodes")
    ap.add_argument("--m7000-density-radius-km", default=M7000_DENSITY_RADIUS_KM, type=float, help="Radius for M7000 local point count on SCHISM nodes")
    ap.add_argument("--m7000-zone-dk-max-km", default=M7000_ZONE_DK_MAX_KM, type=float, help="d_k threshold for M7000-trusted zone")
    ap.add_argument("--etopo-zone-dk-min-km", default=ETOPO_ZONE_DK_MIN_KM, type=float, help="d_k threshold for ETOPO-trusted zone")
    ap.add_argument("--diff-cbar-lims", nargs=2, type=float, default=DIFF_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Manual colorbar limits for diff map")
    ap.add_argument("--depth-cbar-lims", nargs=2, type=float, default=DEPTH_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Shared colorbar limits for M7000/ETOPO depth comparison")
    ap.add_argument("--roughness-cbar-lims", nargs=2, type=float, default=ROUGHNESS_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Shared colorbar limits for roughness comparison maps")
    ap.add_argument("--slope-cbar-lims", nargs=2, type=float, default=SLOPE_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Shared colorbar limits for slope proxy comparison maps")
    ap.add_argument("--density-dist-cbar-lims", nargs=2, type=float, default=DENSITY_DIST_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Shared colorbar limits for M7000 density distance maps (d1 and d_k)")
    ap.add_argument("--density-count-cbar-lims", nargs=2, type=float, default=DENSITY_COUNT_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Colorbar limits for M7000 local point-count map")
    ap.add_argument("--diff-cmap", default=DIFF_CMAP, help="Matplotlib colormap for difference map")
    ap.add_argument("--depth-cmap", default=DEPTH_CMAP, help="Matplotlib colormap for side-by-side depth maps")
    ap.add_argument("--roughness-cmap", default=ROUGHNESS_CMAP, help="Matplotlib colormap for roughness comparison maps")
    ap.add_argument("--slope-cmap", default=SLOPE_CMAP, help="Matplotlib colormap for slope proxy comparison maps")
    ap.add_argument("--density-dist-cmap", default=DENSITY_DIST_CMAP, help="Matplotlib colormap for M7000 density distance maps")
    ap.add_argument("--density-count-cmap", default=DENSITY_COUNT_CMAP, help="Matplotlib colormap for M7000 local point-count map")
    args = ap.parse_args()

    gd, gx, gy = _load_schism_nodes(args.gr3)
    n_nodes = gx.size
    print(f"Loaded SCHISM grid nodes: {n_nodes}")

    if args.plot_etopo_check:
        _plot_etopo_check(gd, args.etopo_tif, args.out_dir, args.out_prefix, args.show)

    m7000_vals, m7000_npz, m7000_layout = _sample_m7000_npz(
        args.m7000_npz,
        gx,
        gy,
        args.m7000_interp,
        args.m7000_key,
        point_interp=args.m7000_point_interp,
        point_k=args.m7000_point_k,
        point_max_dist_km=args.m7000_point_max_dist_km,
    )
    print(f"Sampled M7000 NPZ to SCHISM nodes (layout={m7000_layout})")

    etopo_vals = _sample_etopo_tif(args.etopo_tif, gx, gy)
    print("Sampled ETOPO tif to SCHISM nodes")

    # Convert ETOPO to depth positive-down if needed
    etopo_depth = etopo_vals.astype(np.float32)
    if args.etopo_is_elevation and args.etopo_depth_sign_flip:
        etopo_depth = (-etopo_depth).astype(np.float32)
    if args.etopo_add_tp_bias_m != 0.0:
        etopo_depth = (etopo_depth + np.float32(args.etopo_add_tp_bias_m)).astype(np.float32)

    valid_m = np.isfinite(m7000_vals)
    valid_e = np.isfinite(etopo_depth)
    overlap = valid_m & valid_e
    diff = np.full(n_nodes, np.nan, dtype=np.float32)
    diff[overlap] = (m7000_vals[overlap] - etopo_depth[overlap]).astype(np.float32)

    print(f"M7000 valid nodes: {int(valid_m.sum())}")
    print(f"ETOPO valid nodes: {int(valid_e.sum())}")
    print(f"Overlap valid nodes: {int(overlap.sum())}")
    if int(valid_e.sum()) == 0:
        print(
            "[WARN] ETOPO valid nodes == 0. This usually indicates a CRS/coordinate mismatch "
            "(e.g., projected SCHISM grid vs lon/lat raster) or an incorrect TIFF path."
        )
        print(
            "       Use the ETOPO check plot/metadata output above to verify raster bounds vs SCHISM grid domain."
        )

    # Point-cloud density diagnostics (trust/region metric for M7000-vs-ETOPO blending)
    density_d1_km = np.full(n_nodes, np.nan, dtype=np.float32)
    density_dk_km = np.full(n_nodes, np.nan, dtype=np.float32)
    density_count_r = np.zeros(n_nodes, dtype=np.int32)
    density_zone_code = np.zeros(n_nodes, dtype=np.int8)
    if m7000_layout == "point":
        try:
            density_d1_km, density_dk_km, density_count_r = _compute_m7000_point_density_metrics(
                px=m7000_npz["lon"],
                py=m7000_npz["lat"],
                qx=gx,
                qy=gy,
                k=args.m7000_density_k,
                radius_km=args.m7000_density_radius_km,
            )
            density_zone_code = _classify_density_zones(
                density_dk_km,
                m7000_dk_max_km=args.m7000_zone_dk_max_km,
                etopo_dk_min_km=args.etopo_zone_dk_min_km,
            )
            print(f"M7000 density diagnostics computed (point-cloud): d1, d{int(args.m7000_density_k)}, count(r={args.m7000_density_radius_km:g} km)")
            _print_stats("M7000 density d1 (km)", density_d1_km)
            _print_stats(f"M7000 density d{int(args.m7000_density_k)} (km)", density_dk_km)
            _print_stats(f"M7000 density count r={args.m7000_density_radius_km:g} km", density_count_r.astype(np.float64))
            print(
                "Density-derived zones:",
                f"M7000={int(np.sum(density_zone_code == 1))}",
                f"blend={int(np.sum(density_zone_code == 2))}",
                f"ETOPO={int(np.sum(density_zone_code == 3))}",
            )
        except Exception as exc:
            print(f"[WARN] Failed to compute M7000 density diagnostics: {exc}")
    else:
        print("M7000 density diagnostics skipped (M7000 NPZ layout is gridded, not point-cloud).")

    adjacency = _build_node_adjacency(gd, n_nodes)
    edge_mask = _m7000_edge_nodes(valid_m, adjacency)
    dist_edge_km = _distance_to_edge_km(gx, gy, edge_mask)
    seam_mask = np.isfinite(dist_edge_km) & (dist_edge_km <= float(args.seam_corridor_km))

    print(f"M7000 coverage edge nodes: {int(edge_mask.sum())}")
    print(f"Seam corridor nodes (<= {args.seam_corridor_km} km): {int(seam_mask.sum())}")

    _print_stats("Diff stats (all overlap)", diff[overlap])
    _print_stats("Diff stats (seam overlap)", diff[overlap & seam_mask])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save node-level comparison arrays
    out_npz = out_dir / f"{args.out_prefix}.npz"
    np.savez_compressed(
        out_npz,
        x=gx.astype(np.float64),
        y=gy.astype(np.float64),
        m7000_depth=m7000_vals.astype(np.float32),
        etopo_depth=etopo_depth.astype(np.float32),
        diff_m7000_minus_etopo=diff.astype(np.float32),
        valid_m7000=valid_m.astype(np.uint8),
        valid_etopo=valid_e.astype(np.uint8),
        overlap=overlap.astype(np.uint8),
        m7000_edge=edge_mask.astype(np.uint8),
        dist_to_m7000_edge_km=dist_edge_km.astype(np.float32),
        seam_corridor=seam_mask.astype(np.uint8),
        seam_corridor_km=np.asarray(float(args.seam_corridor_km)),
        m7000_layout=np.asarray(m7000_layout),
        m7000_point_interp=np.asarray(str(args.m7000_point_interp)),
        m7000_point_k=np.asarray(int(args.m7000_point_k)),
        m7000_point_max_dist_km=np.asarray(
            np.nan if args.m7000_point_max_dist_km is None else float(args.m7000_point_max_dist_km)
        ),
        m7000_density_k=np.asarray(int(args.m7000_density_k)),
        m7000_density_radius_km=np.asarray(float(args.m7000_density_radius_km)),
        m7000_density_d1_km=density_d1_km.astype(np.float32),
        m7000_density_dk_km=density_dk_km.astype(np.float32),
        m7000_density_count_r=density_count_r.astype(np.int32),
        density_zone_code=density_zone_code.astype(np.int8),
        m7000_zone_dk_max_km=np.asarray(float(args.m7000_zone_dk_max_km)),
        etopo_zone_dk_min_km=np.asarray(float(args.etopo_zone_dk_min_km)),
        etopo_add_tp_bias_m=np.asarray(float(args.etopo_add_tp_bias_m)),
        note=np.asarray(
            "ETOPO comparison may include vertical datum mismatch unless ETOPO is converted to TP."
        ),
    )
    print("Saved comparison NPZ:", out_npz)

    rows = []
    for label, arr in [
        ("all_overlap", diff[overlap]),
        ("seam_overlap", diff[overlap & seam_mask]),
    ]:
        s = _stats(arr)
        if s is None:
            rows.append({"label": label, "count": 0, "min": "", "p05": "", "median": "", "mean": "", "p95": "", "max": "", "std": ""})
        else:
            s["label"] = label
            rows.append(s)
    if m7000_layout == "point":
        for label, arr in [
            ("m7000_density_d1_km_all", density_d1_km),
            (f"m7000_density_d{int(args.m7000_density_k)}_km_all", density_dk_km),
            (f"m7000_density_count_r{args.m7000_density_radius_km:g}km_all", density_count_r.astype(np.float64)),
            ("m7000_density_dk_km_seam", density_dk_km[seam_mask]),
        ]:
            s = _stats(arr)
            if s is None:
                rows.append({"label": label, "count": 0, "min": "", "p05": "", "median": "", "mean": "", "p95": "", "max": "", "std": ""})
            else:
                s["label"] = label
                rows.append(s)
    out_csv = out_dir / f"{args.out_prefix}_stats.csv"
    _write_stats_csv(out_csv, rows)
    print("Saved stats CSV:", out_csv)

    if args.plot:
        if args.plot_depth_compare:
            _plot_depth_compare_maps(
                gd=gd,
                x=gx,
                y=gy,
                m7000_depth=m7000_vals,
                etopo_depth=etopo_depth,
                valid_m7000=valid_m,
                valid_etopo=valid_e,
                out_dir=out_dir,
                prefix=args.out_prefix,
                show=args.show,
                max_plot_points=args.max_plot_points,
                cbar_lims=args.depth_cbar_lims,
                depth_cmap=args.depth_cmap,
            )
        if args.plot_m7000_density and m7000_layout == "point":
            _plot_m7000_density_diagnostics(
                gd=gd,
                x=gx,
                y=gy,
                d1_km=density_d1_km,
                dk_km=density_dk_km,
                count_r=density_count_r,
                zone_code=density_zone_code,
                k=int(args.m7000_density_k),
                radius_km=float(args.m7000_density_radius_km),
                m7000_zone_dk_max_km=float(args.m7000_zone_dk_max_km),
                etopo_zone_dk_min_km=float(args.etopo_zone_dk_min_km),
                out_dir=out_dir,
                prefix=args.out_prefix,
                show=args.show,
                max_plot_points=args.max_plot_points,
                dist_cbar_lims=args.density_dist_cbar_lims,
                count_cbar_lims=args.density_count_cbar_lims,
                dist_cmap=args.density_dist_cmap,
                count_cmap=args.density_count_cmap,
            )
        if args.plot_roughness_slope:
            _plot_roughness_slope_diagnostics(
                gd=gd,
                x=gx,
                y=gy,
                m7000_depth=m7000_vals,
                etopo_depth=etopo_depth,
                valid_m7000=valid_m,
                valid_etopo=valid_e,
                adjacency=adjacency,
                out_dir=out_dir,
                prefix=args.out_prefix,
                show=args.show,
                max_plot_points=args.max_plot_points,
                roughness_cbar_lims=args.roughness_cbar_lims,
                slope_cbar_lims=args.slope_cbar_lims,
                roughness_cmap=args.roughness_cmap,
                slope_cmap=args.slope_cmap,
            )
        _plot_maps(
            gd=gd,
            x=gx,
            y=gy,
            diff=diff,
            overlap_mask=overlap,
            seam_mask=seam_mask,
            edge_mask=edge_mask,
            out_dir=out_dir,
            prefix=args.out_prefix,
            show=args.show,
            max_plot_points=args.max_plot_points,
            cbar_lims=args.diff_cbar_lims,
            diff_cmap=args.diff_cmap,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
