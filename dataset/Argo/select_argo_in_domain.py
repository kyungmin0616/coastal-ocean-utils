#!/usr/bin/env python3
"""Select Argo profile NetCDF files inside a SCHISM grid domain.

This script scans Argo NetCDF profile files, checks whether any profile
location falls inside the SCHISM grid polygon (hgrid.gr3), and optionally
copies matching files to a separate folder.

Examples:
  python select_argo_in_domain.py --argo-dir ./2022 --grid hgrid.gr3
  python select_argo_in_domain.py --argo-dir ./2022 --grid hgrid.gr3 \
      --copy --out-dir ./argo_in_domain
  python select_argo_in_domain.py --argo-dir ./2022 --grid hgrid.gr3 \
      --pattern "*.nc" --recursive --wrap-lon
"""

import argparse
import glob as glob_module
import os
import shutil
import sys

from pylib import *  # noqa: F403


LAT_NAMES = ("LATITUDE", "latitude", "lat", "profile_latitude")
LON_NAMES = ("LONGITUDE", "longitude", "lon", "profile_longitude")


def _find_nc_files(argo_dir, pattern="*.nc", recursive=False):
    if recursive:
        search = os.path.join(argo_dir, "**", pattern)
        return sorted(glob_module.glob(search, recursive=True))
    search = os.path.join(argo_dir, pattern)
    return sorted(glob_module.glob(search))


def _read_var(C, names, nc_path):
    for name in names:
        if name in C.variables:
            var = C.variables[name]
            vals = array(var[:], dtype=float).ravel()  # noqa: F403
            fill = getattr(var, "_FillValue", None)
            if fill is None:
                fill = getattr(var, "missing_value", None)
            if fill is not None:
                vals = where(vals == float(fill), nan, vals)  # noqa: F403
            return vals
    raise KeyError("None of {} found in {}".format(names, nc_path))


def _get_lat_lon(nc_path, wrap_lon=False):
    C = ReadNC(nc_path, 1)  # noqa: F403
    try:
        lats = _read_var(C, LAT_NAMES, nc_path)
        lons = _read_var(C, LON_NAMES, nc_path)
    finally:
        try:
            C.close()
        except Exception:
            pass
    if lats.size != lons.size:
        raise ValueError(
            "lat/lon size mismatch in {}: {} vs {}".format(nc_path, lats.size, lons.size)
        )
    if wrap_lon:
        lons = where(lons > 180.0, lons - 360.0, lons)  # noqa: F403
    return lats, lons


def _select_inside_grid(gd, nc_files, wrap_lon=False, verbose=False):
    selected = []
    failed = []
    for nc_path in nc_files:
        try:
            lats, lons = _get_lat_lon(nc_path, wrap_lon=wrap_lon)
            ok = isfinite(lats) & isfinite(lons)  # noqa: F403
            if not ok.any():
                continue
            pts = c_[lons[ok], lats[ok]]  # noqa: F403
            inside = gd.inside_grid(pts)
            if (inside == 1).any():
                selected.append(nc_path)
                if verbose:
                    print("[INFO] inside grid: {}".format(nc_path))
        except Exception as e:
            failed.append((nc_path, str(e)))
            if verbose:
                print("[WARN] failed {}: {}".format(nc_path, e))
    return selected, failed


def _copy_files(selected, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for src in selected:
        dst = os.path.join(out_dir, os.path.basename(src))
        shutil.copy2(src, dst)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Find Argo profile files inside a SCHISM grid domain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--argo-dir", required=True, help="Directory with Argo NetCDF files.")
    parser.add_argument("--grid", required=True, help="Path to SCHISM hgrid.gr3 (or .npz).")
    parser.add_argument("--pattern", default="*.nc", help="Glob pattern for NetCDF files.")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories.")
    parser.add_argument("--wrap-lon", action="store_true", help="Convert 0-360 lon to -180..180.")
    parser.add_argument("--copy", action="store_true", help="Copy selected files to --out-dir.")
    parser.add_argument("--out-dir", default="argo_in_domain", help="Output directory for copies.")
    parser.add_argument("--report", help="Write selected file list to this text file.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file status.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    argo_dir = os.path.expanduser(args.argo_dir)
    if not os.path.isdir(argo_dir):
        print("Error: argo-dir {} not found".format(argo_dir), file=sys.stderr)
        return 1

    grid_path = os.path.expanduser(args.grid)
    if not os.path.exists(grid_path):
        print("Error: grid {} not found".format(grid_path), file=sys.stderr)
        return 1

    gd = loadz(grid_path).hgrid if grid_path.endswith(".npz") else read_schism_hgrid(grid_path)  # noqa: F403

    nc_files = _find_nc_files(argo_dir, pattern=args.pattern, recursive=args.recursive)
    if not nc_files:
        print("No NetCDF files found in {}".format(argo_dir))
        return 0

    selected, failed = _select_inside_grid(
        gd, nc_files, wrap_lon=args.wrap_lon, verbose=args.verbose
    )

    print("Total files scanned: {}".format(len(nc_files)))
    print("Inside grid: {}".format(len(selected)))
    if failed:
        print("Failed to parse: {}".format(len(failed)))
        for path, msg in failed[:5]:
            print("  - {} ({})".format(path, msg))
        if len(failed) > 5:
            print("  ...")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            for path in selected:
                f.write("{}\n".format(path))
        print("Wrote report to {}".format(args.report))

    if args.copy:
        out_dir = os.path.expanduser(args.out_dir)
        _copy_files(selected, out_dir)
        print("Copied {} files to {}".format(len(selected), out_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
