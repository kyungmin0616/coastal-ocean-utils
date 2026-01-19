#!/usr/bin/env python3
"""
Modify SCHISM bathymetry using regions.gr3 with gradual blending across boundaries.

Usage examples:
  python change_dp.py hgrid.gr3 regions.gr3 --output hgrid_changed.gr3 --blend-dist 5000
  python change_dp.py hgrid.gr3 regions.gr3 --region-values 1:-5 2:-10 --blend-dist 2000
  python change_dp.py hgrid.gr3 regions.gr3 --mode add --blend-dist 1000
"""
import argparse
import os
import sys

import numpy as np
from pylib import near_pts, read_schism_hgrid


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


def _blend_weights(dist, max_dist, mode="cosine"):
    w = np.clip(dist / max_dist, 0.0, 1.0)
    if mode == "cosine":
        w = 0.5 * (1.0 - np.cos(np.pi * w))
    return w


def _parse_region_values(entries):
    mapping = {}
    for entry in entries:
        for item in str(entry).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                key, val = item.split(":", 1)
            elif "=" in item:
                key, val = item.split("=", 1)
            else:
                raise ValueError(f"Invalid region mapping '{item}'")
            mapping[int(float(key))] = float(val)
    return mapping


def change_bathymetry(
    hgrid_path,
    regions_path,
    output_path,
    blend_dist,
    blend_mode,
    default_value,
    dist_mode,
    region_values,
    mode,
):
    gd = read_schism_hgrid(hgrid_path)
    rg = read_schism_hgrid(regions_path)
    if gd.np != rg.np:
        raise ValueError("hgrid and regions.gr3 node counts do not match")

    base_dp = gd.dp.copy()
    region_map = _parse_region_values(region_values or [])

    if region_map:
        labels = np.rint(rg.dp).astype(int)
        region_ids = np.array(sorted(region_map.keys()), dtype=int)
        mask_change = np.isin(labels, region_ids)
        target_val = np.full(gd.np, default_value, dtype=float)
        for rid, val in region_map.items():
            target_val[labels == rid] = val
    else:
        labels = rg.dp.copy()
        mask_change = labels != default_value
        target_val = rg.dp.copy()

    if not np.any(mask_change):
        print("No nodes selected for change; writing original hgrid.")
        gd.write_hgrid(output_path)
        return

    if mode == "add":
        target_val[mask_change] = base_dp[mask_change] + target_val[mask_change]

    if dist_mode == "auto":
        dist_mode = "meters" if _is_lonlat(gd.x, gd.y) else "xy"

    if blend_dist <= 0:
        gd.dp = base_dp
        gd.dp[mask_change] = target_val[mask_change]
        gd.write_hgrid(output_path)
        print(f"Wrote {output_path}")
        return

    min_dist = np.full(gd.np, np.inf, dtype=float)
    neighbor_idx = np.full(gd.np, -1, dtype=int)

    xy = np.c_[gd.x, gd.y]
    unique_labels = np.unique(labels[mask_change])

    for label in unique_labels:
        idx_region = np.where(labels == label)[0]
        idx_other = np.where(labels != label)[0]
        if idx_region.size == 0 or idx_other.size == 0:
            continue
        pts_region = xy[idx_region]
        pts_other = xy[idx_other]
        nn = near_pts(pts_region, pts_other)
        nei = idx_other[nn]

        if dist_mode == "xy":
            dist = np.hypot(
                gd.x[idx_region] - gd.x[nei],
                gd.y[idx_region] - gd.y[nei],
            )
        else:
            dist = _haversine_dist(
                gd.x[idx_region],
                gd.y[idx_region],
                gd.x[nei],
                gd.y[nei],
            )

        update = dist < min_dist[idx_region]
        min_dist[idx_region[update]] = dist[update]
        neighbor_idx[idx_region[update]] = nei[update]

    final_dp = base_dp.copy()
    has_neighbor = mask_change & (neighbor_idx >= 0)
    if not np.any(has_neighbor):
        final_dp[mask_change] = target_val[mask_change]
    else:
        dist = min_dist[has_neighbor]
        w = _blend_weights(dist, blend_dist, blend_mode)
        nei = neighbor_idx[has_neighbor]
        neighbor_in_change = mask_change[nei]
        neighbor_val = np.where(neighbor_in_change, target_val[nei], base_dp[nei])
        final_dp[has_neighbor] = w * target_val[has_neighbor] + (1.0 - w) * neighbor_val
        final_dp[mask_change & ~has_neighbor] = target_val[mask_change & ~has_neighbor]

    gd.dp = final_dp
    gd.write_hgrid(output_path)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Modify bathymetry using regions.gr3 with blending across regions."
    )
    parser.add_argument("hgrid", help="Path to source hgrid.gr3.")
    parser.add_argument("regions", help="Path to regions.gr3.")
    parser.add_argument(
        "--output",
        "-o",
        default="hgrid_changed.gr3",
        help="Output grid file (default: hgrid_changed.gr3).",
    )
    parser.add_argument(
        "--blend-dist",
        type=float,
        default=0.0,
        help="Blend distance in meters (lon/lat) or map units (xy).",
    )
    parser.add_argument(
        "--blend-mode",
        choices=["linear", "cosine"],
        default="cosine",
        help="Blending function (default: cosine).",
    )
    parser.add_argument(
        "--default",
        type=float,
        default=0.0,
        help="Default value in regions.gr3 indicating no change.",
    )
    parser.add_argument(
        "--dist-mode",
        choices=["auto", "xy", "meters"],
        default="auto",
        help="Distance mode for blending (default: auto).",
    )
    parser.add_argument(
        "--region-values",
        action="append",
        help=(
            "Mapping of region id to target depth, e.g., 1:-5 or 1=-5. "
            "If provided, regions.gr3 is treated as region ids."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["set", "add"],
        default="set",
        help="Apply region values as absolute depths (set) or offsets (add).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.hgrid):
        print(f"Missing hgrid file: {args.hgrid}", file=sys.stderr)
        return 1
    if not os.path.exists(args.regions):
        print(f"Missing regions file: {args.regions}", file=sys.stderr)
        return 1

    change_bathymetry(
        args.hgrid,
        args.regions,
        args.output,
        args.blend_dist,
        args.blend_mode,
        args.default,
        args.dist_mode,
        args.region_values,
        args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
