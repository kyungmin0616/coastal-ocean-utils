#!/usr/bin/env python3
from __future__ import annotations

"""
Convert M7000 NPZ (gridded or point-cloud) to 3-column text:

  lon  lat  depth

Notes:
- Default data key is 'elev' for compatibility with m2npz/m2npz_vdatum outputs.
- For m2npz_vdatum.py output, 'elev' contains TP-referenced depth (positive-down).
"""

import argparse
from pathlib import Path

import numpy as np


# ----------------------------
# Config (CLI overrides)
# ----------------------------
INPUT_NPZ = "/Users/kpark/Downloads/M7005_TP.npz"
OUTPUT_TXT = ""  # default: <input_stem>_lonlatdepth.txt next to input
DATA_KEY = "elev"
DELIMITER = " "
HEADER = False
FMT = "%.10f %.10f %.6f"
SKIP_NAN = True


def _infer_output_path(in_path: str) -> str:
    p = Path(in_path)
    return str(p.with_name(f"{p.stem}_lonlatdepth.txt"))


def _flatten_npz_lonlatz(z: np.lib.npyio.NpzFile, data_key: str):
    if "lon" not in z or "lat" not in z or data_key not in z:
        missing = [k for k in ("lon", "lat", data_key) if k not in z]
        raise SystemExit(f"NPZ missing required keys: {missing}")

    lon = np.asarray(z["lon"])
    lat = np.asarray(z["lat"])
    val = np.asarray(z[data_key])

    # Gridded layout: lon 1D, lat 1D, value 2D [lat, lon]
    if lon.ndim == 1 and lat.ndim == 1 and val.ndim == 2:
        if val.shape != (lat.size, lon.size):
            raise SystemExit(
                f"Grid shape mismatch: {data_key}.shape={val.shape} expected ({lat.size}, {lon.size})"
            )
        lon2, lat2 = np.meshgrid(lon, lat)
        return lon2.ravel(), lat2.ravel(), val.ravel(), "grid"

    # Point-cloud layout: lon/lat/value all 1D same length
    if lon.ndim == 1 and lat.ndim == 1 and val.ndim == 1:
        if not (lon.size == lat.size == val.size):
            raise SystemExit(
                f"Point-cloud length mismatch: lon={lon.size}, lat={lat.size}, {data_key}={val.size}"
            )
        return lon.astype(float), lat.astype(float), val.astype(float), "point"

    raise SystemExit(
        f"Unsupported NPZ layout: lon.ndim={lon.ndim}, lat.ndim={lat.ndim}, {data_key}.ndim={val.ndim}"
    )


def main():
    ap = argparse.ArgumentParser(description="Convert M7000 NPZ to lon/lat/depth text file.")
    ap.add_argument("--input", default=INPUT_NPZ, help="Input NPZ path")
    ap.add_argument("--output", default=OUTPUT_TXT, help="Output text path (default: <input>_lonlatdepth.txt)")
    ap.add_argument("--key", default=DATA_KEY, help="NPZ data key (default: elev)")
    ap.add_argument("--delimiter", default=DELIMITER, help="Column delimiter (default: space)")
    ap.add_argument("--header", default=HEADER, action=argparse.BooleanOptionalAction, help="Write header line: lon lat depth")
    ap.add_argument("--fmt", default=FMT, help="numpy.savetxt format string")
    ap.add_argument("--skip-nan", default=SKIP_NAN, action=argparse.BooleanOptionalAction, help="Drop rows with NaN in lon/lat/value")
    args = ap.parse_args()

    in_npz = Path(args.input)
    out_txt = Path(args.output) if args.output else Path(_infer_output_path(args.input))
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    z = np.load(in_npz, allow_pickle=True)
    lon, lat, depth, layout = _flatten_npz_lonlatz(z, args.key)

    arr = np.column_stack([lon, lat, depth]).astype(np.float64, copy=False)
    if args.skip_nan:
        m = np.isfinite(arr).all(axis=1)
        dropped = int(arr.shape[0] - np.sum(m))
        arr = arr[m]
    else:
        dropped = 0

    header = "lon lat depth" if args.header else ""
    np.savetxt(
        out_txt,
        arr,
        fmt=args.fmt,
        delimiter=args.delimiter,
        header=header,
        comments="",
    )

    print(f"Input NPZ      : {in_npz}")
    print(f"Layout         : {layout}")
    print(f"Data key       : {args.key}")
    print(f"Rows written   : {arr.shape[0]}")
    print(f"Rows dropped   : {dropped} (NaN rows)" if args.skip_nan else "Rows dropped   : 0 (skip-nan disabled)")
    print(f"Output text    : {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

