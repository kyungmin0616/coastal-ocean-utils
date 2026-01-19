#!/usr/bin/env python3
"""
Modify bathymetry using multiple *.reg regions, with smooth transitions
across region boundaries and optional before/after plots.
"""
from pylib import *
import numpy as np

# -----------------------------------------------------------------------------
# inputs (edit here)
# -----------------------------------------------------------------------------
# source grid and output grid
hgrid_in = 'hgrid.ll.new'
hgrid_out = 'hgrid.ll'

# regions and values (order matters)
regions = [
    'min_h_harlem.reg',
    'min_h_BronxKill.reg',
    'min_h_Hudson.reg',
    'min_h_Passaic_Hackensack.reg',
    'min_h_Rah.reg',
    'min_h_rari.reg',
    'min_h_south_bound.reg',
    'min_h_north_bound.reg',
    'tri1.reg',
    'tri2_permont.reg',
    'min_h_north_bound2.reg',
    'min_h_north_bound3.reg',
    'min_h_north_bound4.reg',
    'min_h_north_bound5.reg',
    'min_h_north_bound6.reg',
]
vals = [0.5, 5, 5, 5, 5, 5, 20, 55, 5, 5, 38, 62, 50, 58, 58]
# mode: 0 => set dp to rvalue when dp<=0; nonzero => enforce min depth (dp<rvalue)
modes = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# blending across boundaries
blend_dist = 2000.0  # meters for lon/lat, or map units for projected coords
blend_mode = 'cosine'  # 'linear' or 'cosine'
dist_mode = 'auto'  # 'auto', 'xy', or 'meters'

# plots
plot_output = 'dp_compare.png'  # set to None to skip plotting


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _is_lonlat(x, y):
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    return xmin >= -360.0 and xmax <= 360.0 and ymin >= -90.0 and ymax <= 90.0


def _haversine_dist(lon1, lat1, lon2, lat2):
    r = 6371000.0
    to_r = np.pi / 180.0
    lon1 = lon1 * to_r
    lat1 = lat1 * to_r
    lon2 = lon2 * to_r
    lat2 = lat2 * to_r
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _blend_weights(dist, max_dist, mode='cosine'):
    w = np.clip(dist / max_dist, 0.0, 1.0)
    if mode == 'cosine':
        w = 0.5 * (1.0 - np.cos(np.pi * w))
    return w


def _assign_regions(gd, regions, vals, modes):
    if not (len(regions) == len(vals) == len(modes)):
        sys.exit('regions/vals/modes must have the same length')

    pts = np.c_[gd.x, gd.y]
    target_dp = gd.dp.copy()
    region_id = np.zeros(gd.np, dtype=int)

    for idx, (region, rvalue, mode) in enumerate(zip(regions, vals, modes), start=1):
        print(f'Processing {region} (value={rvalue}, mode={mode})')
        bp = read_schism_bpfile(region, fmt=1)
        mask = inside_polygon(pts, bp.x, bp.y).astype(bool)
        if not np.any(mask):
            print(f'  No nodes found inside {region}')
            continue

        dp = target_dp[mask].copy()
        if int(mode) == 0:
            fpt = dp <= 0
            dp[fpt] = rvalue
        else:
            fpt = dp < rvalue
            dp[fpt] = rvalue
        target_dp[mask] = dp
        region_id[mask] = idx
        print(f'  Updated {mask.sum()} nodes')

    return target_dp, region_id


def _blend_regions(gd, base_dp, target_dp, region_id, blend_dist, blend_mode, dist_mode):
    final_dp = base_dp.copy()
    mask_region = region_id > 0
    if not np.any(mask_region):
        return final_dp

    if blend_dist <= 0:
        final_dp[mask_region] = target_dp[mask_region]
        return final_dp

    if dist_mode == 'auto':
        dist_mode = 'meters' if _is_lonlat(gd.x, gd.y) else 'xy'

    xy = np.c_[gd.x, gd.y]
    min_dist = np.full(gd.np, np.inf, dtype=float)
    neighbor_idx = np.full(gd.np, -1, dtype=int)

    for rid in np.unique(region_id[mask_region]):
        idx_r = np.where(region_id == rid)[0]
        idx_o = np.where(region_id != rid)[0]
        if idx_r.size == 0 or idx_o.size == 0:
            continue
        nn = near_pts(xy[idx_r], xy[idx_o])
        nei = idx_o[nn]
        if dist_mode == 'xy':
            dist = np.hypot(gd.x[idx_r] - gd.x[nei], gd.y[idx_r] - gd.y[nei])
        else:
            dist = _haversine_dist(
                gd.x[idx_r], gd.y[idx_r], gd.x[nei], gd.y[nei]
            )
        update = dist < min_dist[idx_r]
        min_dist[idx_r[update]] = dist[update]
        neighbor_idx[idx_r[update]] = nei[update]

    has_neighbor = mask_region & (neighbor_idx >= 0)
    if not np.any(has_neighbor):
        final_dp[mask_region] = target_dp[mask_region]
        return final_dp

    dist = min_dist[has_neighbor]
    w = _blend_weights(dist, blend_dist, blend_mode)
    nei = neighbor_idx[has_neighbor]
    neighbor_val = np.where(mask_region[nei], target_dp[nei], base_dp[nei])
    final_dp[has_neighbor] = w * target_dp[has_neighbor] + (1.0 - w) * neighbor_val
    final_dp[mask_region & ~has_neighbor] = target_dp[mask_region & ~has_neighbor]

    return final_dp


def plot_compare(gd, base_dp, new_dp, out_png):
    import matplotlib.pyplot as plt

    vmin = float(np.nanmin([np.nanmin(base_dp), np.nanmin(new_dp)]))
    vmax = float(np.nanmax([np.nanmax(base_dp), np.nanmax(new_dp)]))
    diff = new_dp - base_dp
    dmax = float(np.nanmax(np.abs(diff))) if diff.size else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    gd.dp = base_dp
    h0 = gd.plot(fmt=1, ax=axes[0], cb=False, clim=[vmin, vmax])
    axes[0].set_title('Before')
    fig.colorbar(h0, ax=axes[0], fraction=0.046, pad=0.04)

    gd.dp = new_dp
    h1 = gd.plot(fmt=1, ax=axes[1], cb=False, clim=[vmin, vmax])
    axes[1].set_title('After')
    fig.colorbar(h1, ax=axes[1], fraction=0.046, pad=0.04)

    gd.dp = diff
    h2 = gd.plot(fmt=1, ax=axes[2], cb=False, clim=[-dmax, dmax])
    axes[2].set_title('After - Before')
    fig.colorbar(h2, ax=axes[2], fraction=0.046, pad=0.04)

    gd.dp = base_dp
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    gd = loadz(hgrid_in).hgrid if hgrid_in.endswith('.npz') else read_schism_hgrid(hgrid_in)
    base_dp = gd.dp.copy()

    target_dp, region_id = _assign_regions(gd, regions, vals, modes)
    new_dp = _blend_regions(
        gd, base_dp, target_dp, region_id, blend_dist, blend_mode, dist_mode
    )

    gd.dp = new_dp
    gd.write_hgrid(fname=hgrid_out)
    print(f'Wrote {hgrid_out}')

    if plot_output:
        plot_compare(gd, base_dp, new_dp, plot_output)
        print(f'Wrote {plot_output}')
