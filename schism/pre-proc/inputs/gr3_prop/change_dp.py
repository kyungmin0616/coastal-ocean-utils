#!/usr/bin/env python3
"""
Modify the bathymetry of a SCHISM hgrid.gr3 file based on a regions.gr3 file,
applying a gradual blending between regions with different target depths.
"""
import argparse
import numpy as np
from pylib import *

def _haversine_dist(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points
    on the earth (specified in decimal degrees).
    """
    r = 6371000.0  # Radius of earth in meters
    to_r = np.pi / 180.0
    lon1_rad = lon1 * to_r
    lat1_rad = lat1 * to_r
    lon2_rad = lon2 * to_r
    lat2_rad = lat2 * to_r

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c

def _blend_weights(dist, max_dist, mode='cosine'):
    """
    Calculate blending weights based on distance.
    Weight is 1.0 at max_dist and smoothly decreases to 0.0 at dist=0.
    """
    # Invert distance so that 0 dist = 0 weight, max_dist dist = 1.0 weight
    w = 1.0 - np.clip(dist / max_dist, 0.0, 1.0)
    if mode == 'cosine':
        # The formula from gen_hotstart used (1-cos)/2 which is for 0->1 transition.
        # To get a 1->0 transition, we use (1+cos)/2.
        w = 0.5 * (1 + np.cos(np.pi * w))
    return w

def change_bathymetry(
    hgrid_path, regions_grid_path, output_path,
    blend_dist, blend_mode, default_value
):
    """
    Modifies bathymetry with gradual changes between regions.
    """
    # --- Read inputs ---
    try:
        gd = read_schism_hgrid(hgrid_path)
        print(f"Successfully read hgrid: {hgrid_path}")
    except Exception as e:
        print(f"Error reading hgrid file {hgrid_path}: {e}")
        return

    try:
        rg = read_schism_hgrid(regions_grid_path)
        print(f"Successfully read regions grid: {regions_grid_path}")
    except Exception as e:
        print(f"Error reading regions grid file {regions_grid_path}: {e}")
        return

    if gd.np != rg.np:
        print("Error: The number of nodes in hgrid and regions_grid must be the same.")
        return

    # --- Initialization ---
    original_dp = gd.dp.copy()
    target_dp = rg.dp
    final_dp = original_dp.copy()

    # Find nodes that need to be changed
    nodes_to_change = np.where(target_dp != default_value)[0]
    if nodes_to_change.size == 0:
        print("No regions found with values different from the default. No changes made.")
        gd.write_hgrid(output_path)
        return

    print(f"Found {nodes_to_change.size} nodes to modify.")

    # --- Precompute nearest neighbor and distance ---
    min_dist = np.full(gd.np, np.inf, dtype=float)
    neighbor_val = np.full(gd.np, np.nan, dtype=float)

    unique_regions = np.unique(target_dp)

    for region_val in unique_regions:
        # Find nodes inside the current region and outside
        idx_region = np.where(target_dp == region_val)[0]
        idx_other = np.where(target_dp != region_val)[0]

        if idx_region.size == 0 or idx_other.size == 0:
            continue
        
        print(f"Calculating distances for region with value {region_val}...")

        # Find nearest 'other' node for each node in the current region
        pts_region = c_[gd.x[idx_region], gd.y[idx_region]]
        pts_other = c_[gd.x[idx_other], gd.y[idx_other]]
        
        nn_indices = near_pts(pts_region, pts_other)
        
        # Calculate haversine distance
        dist = _haversine_dist(
            gd.x[idx_region], gd.y[idx_region],
            gd.x[idx_other[nn_indices]], gd.y[idx_other[nn_indices]]
        )

        # Update the global minimum distance and neighbor value
        # This approach ensures we always have the distance to the CLOSEST differing region
        update_mask = dist < min_dist[idx_region]
        min_dist[idx_region[update_mask]] = dist[update_mask]
        neighbor_val[idx_region[update_mask]] = target_dp[idx_other[nn_indices[update_mask]]]

    # --- Blending ---
    # We only blend the nodes that we intended to change
    nodes_with_finite_dist = np.where(np.isfinite(min_dist[nodes_to_change]))[0]
    if nodes_with_finite_dist.size == 0:
        print("Could not find any neighbors for blending. Applying values directly.")
        final_dp[nodes_to_change] = target_dp[nodes_to_change]
    else:
        # Indices relative to the full grid
        blend_nodes_idx = nodes_to_change[nodes_with_finite_dist]

        print(f"Blending {len(blend_nodes_idx)} nodes...")
        
        # Calculate blending weights for these nodes.
        # Note: The logic from gen_hotstart is inverted. There, weight=1 is far from boundary.
        # Here we want weight=1 to be the target value *at* the node.
        # The logic should be: final = w * target_at_node + (1-w) * target_of_neighbor
        dist = min_dist[blend_nodes_idx]
        
        # Weight is high when dist is large, low when dist is small
        w = np.clip(dist / blend_dist, 0.0, 1.0)
        if blend_mode == 'cosine':
            w = 0.5 * (1.0 - np.cos(np.pi * w))
        
        own_values = target_dp[blend_nodes_idx]
        neighbor_values = neighbor_val[blend_nodes_idx]

        # Apply weighted average
        blended_dp = w * own_values + (1 - w) * neighbor_values
        final_dp[blend_nodes_idx] = blended_dp
        
        # For nodes outside blend_dist, directly set the value (where w=1)
        nodes_outside_blend = np.where(min_dist[nodes_to_change] >= blend_dist)[0]
        final_dp[nodes_to_change[nodes_outside_blend]] = target_dp[nodes_to_change[nodes_outside_blend]]

    # --- Write output ---
    gd.dp = final_dp
    try:
        gd.write_hgrid(output_path)
        print(f"Successfully wrote modified hgrid: {output_path}")
    except Exception as e:
        print(f"Error writing output file {output_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Modify hgrid.gr3 bathymetry based on a regions grid, with blending."
    )
    parser.add_argument(
        'hgrid',
        type=str,
        help="Path to the source hgrid.gr3 file to modify."
    )
    parser.add_argument(
        'regions_grid',
        type=str,
        help="Path to the regions.gr3 file containing target depth values."
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='hgrid_changed.gr3',
        help="Name of the output grid file (default: hgrid_changed.gr3)."
    )
    parser.add_argument(
        '--blend_dist',
        type=float,
        required=True,
        help="Blending distance in meters. The bathymetry will be gradually "
             "changed over this distance from a region's boundary."
    )
    parser.add_argument(
        '--blend_mode',
        type=str,
        choices=['linear', 'cosine'],
        default='cosine',
        help="Blending method (default: cosine)."
    )
    parser.add_argument(
        '--default_value',
        type=float,
        default=0.0,
        help="Value in the regions grid that marks areas to be left "
             "unchanged (default: 0.0)."
    )
    args = parser.parse_args()

    change_bathymetry(
        args.hgrid, args.regions_grid, args.output,
        args.blend_dist, args.blend_mode, args.default_value
    )

if __name__ == '__main__':
    main()
