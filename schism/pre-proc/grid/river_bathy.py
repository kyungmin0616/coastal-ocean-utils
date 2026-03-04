#!/usr/bin/env python3
"""
Build river bathymetry on a SCHISM grid from point-based river bed elevations.

Workflow
1) Read all river station CSVs (lon/lat + main/deep bed elevations relative to T.P.)
2) For each *.reg river region, use only station points inside the region
3) Interpolate station values to grid nodes inside the region (nearest station)
4) Smooth the interpolated field along the river (region-restricted graph smoothing)
5) Blend with existing bathymetry near region boundaries (trust existing bathy)

Notes
- Datum/sign convention: station z_bed(T.P.) -> SCHISM depth via dp = -z_bed
- Bathy mode options:
    * 'main'  : use main-channel bed elevation only
    * 'deep'  : use deepest-point bed elevation only
    * 'main_default_deep_limit' : use main as baseline and clamp by deep as a limit
"""

import csv
import builtins
import glob as globlib
import os
import re
import sys
from collections import Counter

import numpy as np

try:
    from pylib import *  # noqa: F401,F403
except Exception:
    # Fallback for local repo layout: <Codes>/{coastal-ocean-utils,pylibs}
    import sys

    _here = os.path.abspath(os.path.dirname(__file__))
    _root = os.path.abspath(os.path.join(_here, "../../../../"))
    sys.path.insert(0, os.path.join(_root, "pylibs"))
    from pylib import *  # noqa: F401,F403


# -----------------------------------------------------------------------------
# inputs (edit here)
# -----------------------------------------------------------------------------
hgrid_in = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/03_1.gr3"
hgrid_out = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/03.gr3"

region_dir = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid"
region_glob = "*.reg"  # use all region files

station_csv_dir = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/MILT_River/bathy"
station_csv_glob = "*.csv"

# Bathy source mode: 'main', 'deep', 'main_default_deep_limit'
bathy_mode = "main_default_deep_limit"

# River smoothing (approximate length scale along grid graph, meters)
smooth_length_m = 500.0
smooth_relax = 0.4       # 0..1, larger => stronger smoothing per iteration
max_smooth_iter = 20     # safety cap
anchor_station_nodes = True

# Blend to existing bathy near region boundary (meters)
boundary_blend_dist_m = 1000.0
blend_mode = "cosine"    # 'linear' or 'cosine'

# Optional: force selected river open boundaries to be artificially wet
# bnd2.bp format is the SCHISM boundary-node list (open boundaries first).
use_open_bnd_wetting = True
open_bnd_bpfile = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/bnd_03_1.bp"
# Keys are 1-based open boundary IDs in bnd2.bp; values are minimum dp (m) at boundary.
open_bnd_wet_depths = {
    2: 5.0,
    3: 5.0,
    4: 5.0,
    4: 5.0,
    5: 5.0,
    6: 5.0,
    7: 5.0,
}
# Smooth inland transition from the selected open boundaries.
open_bnd_wet_transition_m = 1500.0
open_bnd_wet_blend_mode = "cosine"
# If True, inland wetting is limited to nodes already modified by river regions
# (selected boundary nodes are always included).
limit_open_bnd_wetting_to_river_regions = True

# Optional plotting
plot_output = None  # set to None to skip


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _haversine_dist(lon1, lat1, lon2, lat2):
    r = 6371000.0
    to_r = np.pi / 180.0
    lon1 = np.asarray(lon1, dtype=float) * to_r
    lat1 = np.asarray(lat1, dtype=float) * to_r
    lon2 = np.asarray(lon2, dtype=float) * to_r
    lat2 = np.asarray(lat2, dtype=float) * to_r
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def _blend_weights(dist, max_dist, mode="cosine"):
    if max_dist <= 0:
        return np.ones_like(dist, dtype=float)
    w = np.clip(np.asarray(dist, dtype=float) / float(max_dist), 0.0, 1.0)
    if mode == "cosine":
        w = 0.5 * (1.0 - np.cos(np.pi * w))
    return w


def _parse_meter_value(v):
    if v is None:
        return np.nan
    s = str(v).strip()
    if s == "":
        return np.nan
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else np.nan


def _parse_first_int(line):
    m = re.search(r"[-+]?\d+", str(line))
    if m is None:
        raise ValueError(f"Could not parse integer from line: {line!r}")
    return int(m.group(0))


def _read_station_csv(fname):
    rows = []
    with open(fname, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r is None:
                continue
            lon = _parse_meter_value(r.get("Longitude"))
            lat = _parse_meter_value(r.get("Latitude"))
            z_main = _parse_meter_value(r.get("Main Channel Bed Elevation (T.P.)"))
            z_deep = _parse_meter_value(r.get("Deepest Point Bed Elevation (T.P.)"))
            if not np.isfinite(lon) or not np.isfinite(lat):
                continue
            if not np.isfinite(z_main) and not np.isfinite(z_deep):
                continue
            rows.append(
                {
                    "station_name": (r.get("Station Name") or "").strip(),
                    "lon": lon,
                    "lat": lat,
                    "z_main_tp": z_main,
                    "z_deep_tp": z_deep,
                    "dp_main": -z_main if np.isfinite(z_main) else np.nan,
                    "dp_deep": -z_deep if np.isfinite(z_deep) else np.nan,
                    "source": os.path.basename(fname),
                }
            )
    return rows


def read_all_station_points(csv_dir, csv_glob="*.csv"):
    files = sorted(globlib.glob(os.path.join(csv_dir, csv_glob)))
    if len(files) == 0:
        sys.exit(f"No station CSV files found: {os.path.join(csv_dir, csv_glob)}")

    rows = []
    for f in files:
        rr = _read_station_csv(f)
        print(f"Loaded {len(rr):4d} stations from {os.path.basename(f)}")
        rows.extend(rr)

    if len(rows) == 0:
        sys.exit("No valid station rows were parsed from CSVs.")

    pts = {
        "lon": np.array([r["lon"] for r in rows], dtype=float),
        "lat": np.array([r["lat"] for r in rows], dtype=float),
        "dp_main": np.array([r["dp_main"] for r in rows], dtype=float),
        "dp_deep": np.array([r["dp_deep"] for r in rows], dtype=float),
        "station_name": np.array([r["station_name"] for r in rows], dtype=object),
        "source": np.array([r["source"] for r in rows], dtype=object),
    }
    print(f"Total valid stations: {len(rows)}")
    return pts


def _collect_region_files(region_dir, region_glob="*.reg"):
    rfiles = sorted(globlib.glob(os.path.join(region_dir, region_glob)))
    if len(rfiles) == 0:
        sys.exit(f"No region files found: {os.path.join(region_dir, region_glob)}")
    return rfiles


def _nearest_interp(node_xy, st_xy, st_val):
    if len(st_val) == 0:
        return np.full(node_xy.shape[0], np.nan, dtype=float)
    ind = near_pts(node_xy, st_xy)
    return np.asarray(st_val, dtype=float)[ind]


def _build_region_neighbor_data(gd, region_idx):
    if region_idx.size == 0:
        return None

    if not hasattr(gd, "indnd"):
        gd.compute_nne(fmt=1)

    region_mask = np.zeros(gd.np, dtype=bool)
    region_mask[region_idx] = True

    g2l = np.full(gd.np, -1, dtype=int)
    g2l[region_idx] = np.arange(region_idx.size, dtype=int)

    nbr_g = np.asarray(gd.indnd[region_idx], dtype=int)
    valid = nbr_g >= 0
    if np.any(valid):
        valid &= region_mask[nbr_g]

    nbr_l = np.full_like(nbr_g, -1)
    if np.any(valid):
        nbr_l[valid] = g2l[nbr_g[valid]]
        valid &= nbr_l >= 0

    dist = np.full(nbr_g.shape, np.nan, dtype=float)
    if np.any(valid):
        rr, cc = np.where(valid)
        dist[rr, cc] = _haversine_dist(
            gd.x[region_idx[rr]],
            gd.y[region_idx[rr]],
            gd.x[nbr_g[rr, cc]],
            gd.y[nbr_g[rr, cc]],
        )

    return {
        "region_idx": region_idx,
        "region_mask": region_mask,
        "g2l": g2l,
        "nbr_g": nbr_g,
        "nbr_l": nbr_l,
        "valid": valid,
        "dist": dist,
    }


def _estimate_smooth_iterations(smooth_len_m, dist, valid, max_iter=20):
    if smooth_len_m is None or smooth_len_m <= 0:
        return 0
    if dist is None or not np.any(valid):
        return 0
    edge_dist = dist[valid]
    edge_dist = edge_dist[np.isfinite(edge_dist) & (edge_dist > 0)]
    if edge_dist.size == 0:
        return 0
    dx = float(np.median(edge_dist))
    if dx <= 0:
        return 0
    # Approximate mapping: one smoothing iteration ~= one local edge scale.
    n_iter = int(np.clip(np.ceil(float(smooth_len_m) / dx), 1, max_iter))
    return n_iter


def _smooth_region_field(
    field_local,
    neighbor_data,
    smooth_len_m=500.0,
    relax=0.6,
    max_iter=20,
    anchor_local=None,
):
    if field_local.size == 0:
        return field_local.copy()

    valid = neighbor_data["valid"]
    dist = neighbor_data["dist"]
    nbr_l = neighbor_data["nbr_l"]

    n_iter = _estimate_smooth_iterations(smooth_len_m, dist, valid, max_iter=max_iter)
    if n_iter <= 0:
        return field_local.copy()

    w = np.zeros_like(dist, dtype=float)
    m = valid & np.isfinite(dist) & (dist > 0)
    w[m] = 1.0 / dist[m]
    sum_w = w.sum(axis=1)
    self_w = np.where(sum_w > 0, sum_w, 1.0)

    vals = np.asarray(field_local, dtype=float).copy()
    for _ in range(n_iter):
        nbr_vals = np.zeros_like(w, dtype=float)
        if np.any(m):
            nbr_vals[m] = vals[nbr_l[m]]
        avg = (self_w * vals + (w * nbr_vals).sum(axis=1)) / (self_w + sum_w)
        vals = (1.0 - relax) * vals + relax * avg
        if anchor_local is not None:
            amask, aval = anchor_local
            vals[amask] = aval[amask]
    return vals


def _make_anchor_values(region_node_xy, station_xy, station_val):
    if region_node_xy.shape[0] == 0 or station_xy.shape[0] == 0:
        return None
    if station_val.size == 0:
        return None
    node_of_station = near_pts(station_xy, region_node_xy)
    n_local = region_node_xy.shape[0]
    sumv = np.zeros(n_local, dtype=float)
    cnt = np.zeros(n_local, dtype=int)
    for iloc, v in zip(node_of_station, station_val):
        if np.isfinite(v):
            sumv[iloc] += v
            cnt[iloc] += 1
    amask = cnt > 0
    if not np.any(amask):
        return None
    aval = np.zeros(n_local, dtype=float)
    aval[amask] = sumv[amask] / cnt[amask]
    return amask, aval


def _blend_region_with_existing(gd, base_dp, region_idx, target_local, blend_dist_m, mode="cosine"):
    out = base_dp[region_idx].copy()
    if region_idx.size == 0:
        return out

    if blend_dist_m is None or blend_dist_m <= 0:
        return np.asarray(target_local, dtype=float).copy()

    region_mask = np.zeros(gd.np, dtype=bool)
    region_mask[region_idx] = True
    out_idx = np.where(~region_mask)[0]
    if out_idx.size == 0:
        return np.asarray(target_local, dtype=float).copy()

    xy_r = np.c_[gd.x[region_idx], gd.y[region_idx]]
    xy_o = np.c_[gd.x[out_idx], gd.y[out_idx]]
    nn = near_pts(xy_r, xy_o)
    nei = out_idx[nn]
    dist = _haversine_dist(gd.x[region_idx], gd.y[region_idx], gd.x[nei], gd.y[nei])
    w = _blend_weights(dist, blend_dist_m, mode)

    target_local = np.asarray(target_local, dtype=float)
    out = w * target_local + (1.0 - w) * base_dp[region_idx]
    return out


def _read_open_boundary_node_lists(fname, np_nodes=None):
    """
    Read open-boundary node ids from a SCHISM boundary-node list file (e.g., bnd2.bp).

    Expected header:
      line1: <nob> = Number of open boundaries
      line2: <total_open_nodes> = Total number of open boundary nodes
    Then repeated blocks:
      line : <nobn_i> = Number of nodes for open boundary i
      next nobn_i lines: node ids (1-based)
    """
    with open(fname, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if len(lines) < 2:
        raise ValueError(f"Boundary file too short: {fname}")

    nob = _parse_first_int(lines[0])
    tot_open = _parse_first_int(lines[1])
    out = {}
    pos = 2

    for ib in range(1, nob + 1):
        if pos >= len(lines):
            raise ValueError(f"Unexpected EOF while reading open boundary {ib} in {fname}")
        nbn = _parse_first_int(lines[pos])
        pos += 1
        if pos + nbn > len(lines):
            raise ValueError(f"Unexpected EOF in node list for open boundary {ib} in {fname}")

        ids = []
        for j in range(nbn):
            ids.append(_parse_first_int(lines[pos + j]))
        pos += nbn

        ids = np.asarray(ids, dtype=int) - 1  # convert to 0-based
        if np_nodes is not None:
            bad = (ids < 0) | (ids >= int(np_nodes))
            if np.any(bad):
                raise ValueError(
                    f"Boundary {ib} in {fname} has node ids outside grid range: "
                    f"{ids[bad][:10] + 1}"
                )
        out[ib] = ids

    nsum = int(builtins.sum(len(v) for v in out.values()))
    if nsum != int(tot_open):
        print(
            f"WARNING: Open-boundary node total mismatch in {os.path.basename(fname)}: "
            f"header={tot_open}, parsed={nsum}"
        )

    return out


def _apply_open_boundary_wetting(
    gd,
    dp_in,
    open_bnd_nodes,
    wet_depth_map,
    transition_m=1500.0,
    mode="cosine",
    candidate_mask=None,
):
    """
    Enforce a smoothly decaying minimum depth from selected open-boundary nodes inland.

    At the selected boundary nodes:
      dp >= configured artificial depth
    Inland within transition_m:
      dp >= decaying depth floor (blended to zero effect at transition distance)
    """
    if not wet_depth_map:
        return dp_in.copy(), np.zeros(gd.np, dtype=bool)

    seed_nodes = []
    seed_depths = []
    for ib, dep in wet_depth_map.items():
        if ib not in open_bnd_nodes:
            print(f"WARNING: open boundary {ib} not found in boundary file; skip")
            continue
        dep = float(dep)
        if dep <= 0:
            print(f"WARNING: non-positive wet depth for boundary {ib}: {dep}; skip")
            continue
        ids = np.asarray(open_bnd_nodes[ib], dtype=int)
        if ids.size == 0:
            continue
        seed_nodes.append(ids)
        seed_depths.append(np.full(ids.size, dep, dtype=float))

    if len(seed_nodes) == 0:
        print("No valid selected open boundaries for wetting; skip")
        return dp_in.copy(), np.zeros(gd.np, dtype=bool)

    seed_nodes = np.concatenate(seed_nodes)
    seed_depths = np.concatenate(seed_depths)

    # If a node is in multiple selected boundaries, use the larger minimum depth.
    uniq, inv = np.unique(seed_nodes, return_inverse=True)
    depth_u = np.zeros(uniq.size, dtype=float)
    np.maximum.at(depth_u, inv, seed_depths)
    seed_nodes = uniq
    seed_depths = depth_u

    if candidate_mask is None:
        candidate_mask = np.ones(gd.np, dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool).copy()
        if candidate_mask.size != gd.np:
            raise ValueError("candidate_mask size mismatch")
    candidate_mask[seed_nodes] = True  # always include selected open-boundary nodes

    idx = np.where(candidate_mask)[0]
    if idx.size == 0:
        return dp_in.copy(), np.zeros(gd.np, dtype=bool)

    xy_c = np.c_[gd.x[idx], gd.y[idx]]
    xy_s = np.c_[gd.x[seed_nodes], gd.y[seed_nodes]]
    nn = near_pts(xy_c, xy_s)
    nearest_seed_nodes = seed_nodes[nn]
    nearest_seed_depth = seed_depths[nn]
    dist = _haversine_dist(gd.x[idx], gd.y[idx], gd.x[nearest_seed_nodes], gd.y[nearest_seed_nodes])

    # Weight to current field increases inland; at boundary use the artificial wet depth.
    # Only enforce inland floor inside transition distance so far-away nodes are untouched.
    wcur = _blend_weights(dist, transition_m, mode)
    depth_floor = (1.0 - wcur) * nearest_seed_depth

    dp_out = dp_in.copy()
    before = dp_out[idx].copy()
    if transition_m is None or transition_m <= 0:
        apply_mask = np.zeros(idx.size, dtype=bool)
    else:
        apply_mask = dist < float(transition_m)
    if np.any(apply_mask):
        ii = idx[apply_mask]
        dp_out[ii] = np.maximum(dp_out[ii], depth_floor[apply_mask])
    changed = np.zeros(gd.np, dtype=bool)
    changed[idx] = np.abs(dp_out[idx] - before) > 0

    # Enforce exact boundary minima regardless of transition formula.
    bbefore = dp_out[seed_nodes].copy()
    dp_out[seed_nodes] = np.maximum(dp_out[seed_nodes], seed_depths)
    changed[seed_nodes] |= np.abs(dp_out[seed_nodes] - bbefore) > 0

    print(
        "Applied open-boundary wetting: "
        f"{len(seed_nodes)} seed nodes, {changed.sum()} changed nodes, "
        f"transition={transition_m:.1f} m"
    )
    return dp_out, changed


def _plot_compare(gd, base_dp, new_dp, out_png):
    import matplotlib.pyplot as plt

    vmin = float(np.nanmin([np.nanmin(base_dp), np.nanmin(new_dp)]))
    vmax = float(np.nanmax([np.nanmax(base_dp), np.nanmax(new_dp)]))
    diff = new_dp - base_dp
    dmax = float(np.nanmax(np.abs(diff))) if diff.size else 1.0
    if not np.isfinite(dmax) or dmax == 0.0:
        dmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    gd.dp = base_dp
    h0 = gd.plot(fmt=1, ax=axes[0], cb=False, clim=[vmin, vmax])
    axes[0].set_title("Before")
    fig.colorbar(h0, ax=axes[0], fraction=0.046, pad=0.04)

    gd.dp = new_dp
    h1 = gd.plot(fmt=1, ax=axes[1], cb=False, clim=[vmin, vmax])
    axes[1].set_title("After")
    fig.colorbar(h1, ax=axes[1], fraction=0.046, pad=0.04)

    gd.dp = diff
    h2 = gd.plot(fmt=1, ax=axes[2], cb=False, clim=[-dmax, dmax])
    axes[2].set_title("After - Before")
    fig.colorbar(h2, ax=axes[2], fraction=0.046, pad=0.04)

    gd.dp = base_dp
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _finalize_bathy(dp_main_local, dp_deep_local, mode):
    mode = str(mode).strip().lower()
    if mode == "main":
        return dp_main_local.copy()
    if mode == "deep":
        return dp_deep_local.copy()
    if mode == "main_default_deep_limit":
        lo = np.fmin(dp_main_local, dp_deep_local)
        hi = np.fmax(dp_main_local, dp_deep_local)
        out = dp_main_local.copy()
        out = np.clip(out, lo, hi)
        return out
    raise ValueError(f"Unsupported bathy_mode: {mode}")


def _value_stats(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n=0"
    return f"n={a.size}, min={a.min():.3f}, max={a.max():.3f}, mean={a.mean():.3f}"


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Reading grid: {hgrid_in}")
    gd = loadz(hgrid_in).hgrid if hgrid_in.endswith(".npz") else read_schism_hgrid(hgrid_in)
    base_dp = gd.dp.copy()

    print("Reading station CSVs...")
    stations = read_all_station_points(station_csv_dir, station_csv_glob)
    all_station_xy = np.c_[stations["lon"], stations["lat"]]

    region_files = _collect_region_files(region_dir, region_glob)
    print(f"Using {len(region_files)} region files")
    for rf in region_files:
        print(f"  - {os.path.basename(rf)}")

    mode_key = str(bathy_mode).strip().lower()

    new_dp = base_dp.copy()
    updated_mask = np.zeros(gd.np, dtype=bool)
    region_num = np.zeros(gd.np, dtype=int)

    pts_grid = np.c_[gd.x, gd.y]
    neighbor_cache_ready = False

    for ir, rfile in enumerate(region_files, start=1):
        rname = os.path.basename(rfile)
        print(f"\nProcessing region {ir}: {rname}")
        bp = read_schism_bpfile(rfile, fmt=1)

        mask_node = inside_polygon(pts_grid, bp.x, bp.y).astype(bool)
        idx_node = np.where(mask_node)[0]
        if idx_node.size == 0:
            print("  No grid nodes inside region; skip")
            continue

        mask_sta = inside_polygon(all_station_xy, bp.x, bp.y).astype(bool)
        idx_sta = np.where(mask_sta)[0]
        if idx_sta.size == 0:
            print("  No station points inside region; skip")
            continue

        region_station_sources = Counter(stations["source"][idx_sta].tolist())
        print(f"  Grid nodes in region : {idx_node.size}")
        print(f"  Stations in region   : {idx_sta.size}")
        print(f"  Station source count : {dict(region_station_sources)}")

        st_xy = all_station_xy[idx_sta]
        st_main = stations["dp_main"][idx_sta]
        st_deep = stations["dp_deep"][idx_sta]

        valid_main = np.isfinite(st_main)
        valid_deep = np.isfinite(st_deep)
        if mode_key in ("main", "main_default_deep_limit") and not np.any(valid_main):
            print("  No valid main-channel elevations in region; skip")
            continue
        if mode_key in ("deep", "main_default_deep_limit") and not np.any(valid_deep):
            print("  No valid deepest-point elevations in region; skip")
            continue

        node_xy = pts_grid[idx_node]

        dp_main_local = _nearest_interp(node_xy, st_xy[valid_main], st_main[valid_main]) if np.any(valid_main) else np.full(idx_node.size, np.nan)
        dp_deep_local = _nearest_interp(node_xy, st_xy[valid_deep], st_deep[valid_deep]) if np.any(valid_deep) else np.full(idx_node.size, np.nan)

        print(f"  Initial dp_main interp: {_value_stats(dp_main_local)}")
        print(f"  Initial dp_deep interp: {_value_stats(dp_deep_local)}")

        if smooth_length_m and smooth_length_m > 0:
            if not neighbor_cache_ready:
                print("  Building grid node connectivity for smoothing (compute_nne)...")
                gd.compute_nne(fmt=1)
                neighbor_cache_ready = True

            nbd = _build_region_neighbor_data(gd, idx_node)

            anchor_main = None
            anchor_deep = None
            if anchor_station_nodes:
                if np.any(valid_main):
                    anchor_main = _make_anchor_values(node_xy, st_xy[valid_main], st_main[valid_main])
                if np.any(valid_deep):
                    anchor_deep = _make_anchor_values(node_xy, st_xy[valid_deep], st_deep[valid_deep])

            if np.any(valid_main):
                dp_main_local = _smooth_region_field(
                    dp_main_local,
                    nbd,
                    smooth_len_m=smooth_length_m,
                    relax=smooth_relax,
                    max_iter=max_smooth_iter,
                    anchor_local=anchor_main,
                )
            if np.any(valid_deep):
                dp_deep_local = _smooth_region_field(
                    dp_deep_local,
                    nbd,
                    smooth_len_m=smooth_length_m,
                    relax=smooth_relax,
                    max_iter=max_smooth_iter,
                    anchor_local=anchor_deep,
                )

            print(f"  Smoothed dp_main     : {_value_stats(dp_main_local)}")
            print(f"  Smoothed dp_deep     : {_value_stats(dp_deep_local)}")

        # Fill missing sides for modes that only need one field.
        if not np.any(np.isfinite(dp_main_local)) and np.any(np.isfinite(dp_deep_local)):
            dp_main_local = dp_deep_local.copy()
        if not np.any(np.isfinite(dp_deep_local)) and np.any(np.isfinite(dp_main_local)):
            dp_deep_local = dp_main_local.copy()

        target_local = _finalize_bathy(dp_main_local, dp_deep_local, mode_key)
        print(f"  Target river bathy   : {_value_stats(target_local)}")

        final_local = _blend_region_with_existing(
            gd,
            base_dp,
            idx_node,
            target_local,
            blend_dist_m=boundary_blend_dist_m,
            mode=blend_mode,
        )
        print(f"  Blended final bathy  : {_value_stats(final_local)}")

        overlap = updated_mask[idx_node]
        if np.any(overlap):
            print(f"  WARNING: {overlap.sum()} nodes overlap previous region(s); later region overwrites")

        new_dp[idx_node] = final_local
        updated_mask[idx_node] = True
        region_num[idx_node] = ir

    # Optional post-process: keep selected river open-boundary nodes artificially wet
    if use_open_bnd_wetting:
        print(f"\nReading open-boundary nodes: {open_bnd_bpfile}")
        try:
            open_bnd_nodes = _read_open_boundary_node_lists(open_bnd_bpfile, np_nodes=gd.np)
            print(f"Found {len(open_bnd_nodes)} open boundaries in boundary file")
            print(f"Selected river open boundaries for wetting: {sorted(open_bnd_wet_depths)}")
            cand_mask = None
            if limit_open_bnd_wetting_to_river_regions:
                cand_mask = updated_mask.copy()
            new_dp, wet_changed = _apply_open_boundary_wetting(
                gd,
                new_dp,
                open_bnd_nodes,
                open_bnd_wet_depths,
                transition_m=open_bnd_wet_transition_m,
                mode=open_bnd_wet_blend_mode,
                candidate_mask=cand_mask,
            )
            if np.any(wet_changed):
                updated_mask |= wet_changed
        except Exception as e:
            print(f"WARNING: open-boundary wetting failed: {e}")

    gd.dp = new_dp
    gd.write_hgrid(fname=hgrid_out)
    print(f"\nWrote {hgrid_out}")
    print(f"Updated nodes: {updated_mask.sum()} / {gd.np}")

    if plot_output:
        try:
            _plot_compare(gd, base_dp, new_dp, plot_output)
            print(f"Wrote {plot_output}")
        except Exception as e:
            print(f"WARNING: plot failed: {e}")
