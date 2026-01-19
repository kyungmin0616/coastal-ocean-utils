#!/usr/bin/env python3
"""
Generate regions.gr3 from multiple *.reg files.

Usage examples:
  python gen_regions.py --hgrid hgrid.gr3 \
    --regions river1.reg:1 river2.reg:2 --output regions.gr3

  python gen_regions.py --hgrid hgrid.gr3 \
    --regions river1.reg river2.reg --start-value 1 --default 0

  python gen_regions.py --hgrid hgrid.gr3 \
    --regions river1.reg,river2.reg
"""
import argparse
import os
import sys

import numpy as np
from pylib import inside_polygon, read_schism_bpfile, read_schism_hgrid


def _parse_regions(entries, start_value):
    regions = []
    next_val = start_value
    for entry in entries:
        for item in str(entry).split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                path, val = item.split(":", 1)
                try:
                    val = float(val)
                except ValueError:
                    raise ValueError(f"Invalid region value in '{item}'")
            else:
                path, val = item, None
            regions.append([path.strip(), val])

    for item in regions:
        if item[1] is None:
            item[1] = next_val
            next_val += 1

    return [(p, v) for p, v in regions]


def main():
    parser = argparse.ArgumentParser(
        description="Generate regions.gr3 from multiple *.reg files."
    )
    parser.add_argument(
        "--hgrid",
        default="hgrid.gr3",
        help="Path to hgrid.gr3 (default: hgrid.gr3).",
    )
    parser.add_argument(
        "--regions",
        action="append",
        required=True,
        help=(
            "Region file or file:value; can be repeated or comma-separated. "
            "If value is omitted, it is auto-assigned starting at --start-value."
        ),
    )
    parser.add_argument(
        "--start-value",
        type=float,
        default=1.0,
        help="Starting value for auto-assigned region values (default: 1).",
    )
    parser.add_argument(
        "--default",
        type=float,
        default=0.0,
        help="Default value outside all regions (default: 0).",
    )
    parser.add_argument(
        "--output",
        default="regions.gr3",
        help="Output regions file name (default: regions.gr3).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.hgrid):
        print(f"Missing hgrid file: {args.hgrid}", file=sys.stderr)
        return 1

    regions = _parse_regions(args.regions, args.start_value)
    for path, _ in regions:
        if not os.path.exists(path):
            print(f"Missing region file: {path}", file=sys.stderr)
            return 1

    gd = read_schism_hgrid(args.hgrid)
    gd.dp[:] = args.default
    pts = np.c_[gd.x, gd.y]

    overlap_mask = np.zeros(gd.np, dtype=bool)
    for path, value in regions:
        bp = read_schism_bpfile(path, fmt=1)
        mask = inside_polygon(pts, bp.x, bp.y) == 1
        count = int(mask.sum())
        overlap = int((overlap_mask & mask).sum())
        if overlap > 0:
            print(f"Warning: {overlap} nodes overlap in {path}; overwriting values")
        gd.dp[mask] = value
        overlap_mask |= mask
        print(f"Region {path} -> value {value}: {count} nodes")

    gd.write_hgrid(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
