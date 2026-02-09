#!/usr/bin/env python3
"""
Compare two sets of SCHISM sflux NetCDF files.

Typical use:
  python comp_sflux.py --dir-a . --dir-b ./serial

Examples:
  # Compare all sflux files in two folders and print summary only
  python comp_sflux.py --dir-a ./mpi_out --dir-b ./serial_out

  # Compare only air files and write detailed CSV report
  python comp_sflux.py \
    --dir-a ./mpi_out \
    --dir-b ./serial_out \
    --pattern "sflux_air_1.*.nc" \
    --report ./comp_sflux_air.csv

  # Restrict variable comparison to selected vars
  python comp_sflux.py \
    --dir-a ./mpi_out \
    --dir-b ./serial_out \
    --vars prmsl spfh stmp uwind vwind
"""

import argparse
import glob
import os
import sys
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class VarResult:
    file_name: str
    var_name: str
    status: str
    shape_a: str
    shape_b: str
    dtype_a: str
    dtype_b: str
    nan_mismatch: int
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    max_rel_diff: float
    allclose: bool


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare sflux files from two directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dir-a", required=True, help="Directory A (reference).")
    ap.add_argument("--dir-b", required=True, help="Directory B (target).")
    ap.add_argument(
        "--pattern",
        default="sflux_*_1.*.nc",
        help="Filename glob pattern used in both directories.",
    )
    ap.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help="Optional variable list to compare. Default: all common variables.",
    )
    ap.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for allclose.")
    ap.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for allclose.")
    ap.add_argument(
        "--report",
        default=None,
        help="Optional CSV output path for detailed per-variable results.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Print less detail (summary only).",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Only compare the first N common files (sorted by filename).",
    )
    ap.add_argument(
        "--strict-filename",
        action="store_true",
        help="Require exact filename match (disable index-based matching).",
    )
    return ap.parse_args(argv)


def discover_files(root: str, pattern: str) -> Dict[str, str]:
    paths = sorted(glob.glob(os.path.join(root, pattern)))
    out = {}
    for p in paths:
        out[os.path.basename(p)] = p
    return out


def canonical_key(fname: str) -> str:
    # Match padding differences, e.g. sflux_air_1.1.nc == sflux_air_1.0001.nc
    m = re.match(r"^(sflux_(?:air|prc)_1)\.(\d+)\.nc$", fname)
    if m:
        return f"{m.group(1)}.{int(m.group(2))}.nc"
    return fname


def build_compare_maps(files: Dict[str, str], strict_filename: bool) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if strict_filename:
        return dict(files)
    for fname, path in files.items():
        key = canonical_key(fname)
        # keep first if duplicates by canonical key
        if key not in out:
            out[key] = path
    return out


def compare_dims(file_name: str, ds_a: xr.Dataset, ds_b: xr.Dataset) -> List[VarResult]:
    out: List[VarResult] = []
    dims_a = dict(ds_a.sizes)
    dims_b = dict(ds_b.sizes)
    if dims_a == dims_b:
        return out
    keys = sorted(set(dims_a) | set(dims_b))
    for k in keys:
        sa = str(dims_a.get(k, "NA"))
        sb = str(dims_b.get(k, "NA"))
        status = "OK" if sa == sb else "DIM_MISMATCH"
        out.append(
            VarResult(
                file_name=file_name,
                var_name=f"<dim:{k}>",
                status=status,
                shape_a=sa,
                shape_b=sb,
                dtype_a="",
                dtype_b="",
                nan_mismatch=0,
                max_abs_diff=np.nan,
                mean_abs_diff=np.nan,
                rmse=np.nan,
                max_rel_diff=np.nan,
                allclose=(status == "OK"),
            )
        )
    return out


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return np.nan


def compare_numeric(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> Tuple[int, float, float, float, float, bool]:
    if a.shape != b.shape:
        return 0, np.nan, np.nan, np.nan, np.nan, False

    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    nan_mismatch = int(np.sum(nan_a ^ nan_b))

    valid = ~(nan_a | nan_b)
    if not np.any(valid):
        # all NaN on both sides (or no valid points)
        return nan_mismatch, 0.0, 0.0, 0.0, 0.0, (nan_mismatch == 0)

    da = a[valid]
    db = b[valid]
    diff = da - db
    abs_diff = np.abs(diff)
    max_abs = _safe_float(np.max(abs_diff))
    mean_abs = _safe_float(np.mean(abs_diff))
    rmse = _safe_float(np.sqrt(np.mean(diff * diff)))
    denom = np.maximum(np.abs(da), np.abs(db))
    rel = np.zeros_like(abs_diff)
    nz = denom > 0
    rel[nz] = abs_diff[nz] / denom[nz]
    max_rel = _safe_float(np.max(rel))
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    return nan_mismatch, max_abs, mean_abs, rmse, max_rel, ok


def compare_nonnumeric(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


def compare_var(file_name: str, var_name: str, a: xr.DataArray, b: xr.DataArray, atol: float, rtol: float) -> VarResult:
    arr_a = np.asarray(a.values)
    arr_b = np.asarray(b.values)
    shape_a = str(arr_a.shape)
    shape_b = str(arr_b.shape)
    dtype_a = str(arr_a.dtype)
    dtype_b = str(arr_b.dtype)

    if arr_a.shape != arr_b.shape:
        return VarResult(
            file_name=file_name,
            var_name=var_name,
            status="SHAPE_MISMATCH",
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            nan_mismatch=0,
            max_abs_diff=np.nan,
            mean_abs_diff=np.nan,
            rmse=np.nan,
            max_rel_diff=np.nan,
            allclose=False,
        )

    numeric = np.issubdtype(arr_a.dtype, np.number) and np.issubdtype(arr_b.dtype, np.number)
    if numeric:
        nan_mismatch, max_abs, mean_abs, rmse, max_rel, ok = compare_numeric(
            arr_a.astype(np.float64), arr_b.astype(np.float64), atol, rtol
        )
    else:
        ok = compare_nonnumeric(arr_a, arr_b)
        nan_mismatch, max_abs, mean_abs, rmse, max_rel = 0, np.nan, np.nan, np.nan, np.nan

    return VarResult(
        file_name=file_name,
        var_name=var_name,
        status=("OK" if ok else "DIFF"),
        shape_a=shape_a,
        shape_b=shape_b,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        nan_mismatch=nan_mismatch,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        rmse=rmse,
        max_rel_diff=max_rel,
        allclose=ok,
    )


def compare_file(path_a: str, path_b: str, file_name: str, vars_keep: Optional[List[str]], atol: float, rtol: float) -> List[VarResult]:
    results: List[VarResult] = []
    with xr.open_dataset(path_a, decode_times=False) as ds_a, xr.open_dataset(path_b, decode_times=False) as ds_b:
        results.extend(compare_dims(file_name, ds_a, ds_b))

        vars_a = set(ds_a.variables)
        vars_b = set(ds_b.variables)
        common = sorted(vars_a & vars_b)
        if vars_keep is not None:
            common = [v for v in common if v in vars_keep]
        for v in common:
            results.append(compare_var(file_name, v, ds_a[v], ds_b[v], atol, rtol))

        missing_in_b = sorted(vars_a - vars_b)
        missing_in_a = sorted(vars_b - vars_a)
        if vars_keep is not None:
            missing_in_b = [v for v in missing_in_b if v in vars_keep]
            missing_in_a = [v for v in missing_in_a if v in vars_keep]

        for v in missing_in_b:
            results.append(
                VarResult(
                    file_name=file_name,
                    var_name=v,
                    status="MISSING_IN_B",
                    shape_a="NA",
                    shape_b="NA",
                    dtype_a="NA",
                    dtype_b="NA",
                    nan_mismatch=0,
                    max_abs_diff=np.nan,
                    mean_abs_diff=np.nan,
                    rmse=np.nan,
                    max_rel_diff=np.nan,
                    allclose=False,
                )
            )
        for v in missing_in_a:
            results.append(
                VarResult(
                    file_name=file_name,
                    var_name=v,
                    status="MISSING_IN_A",
                    shape_a="NA",
                    shape_b="NA",
                    dtype_a="NA",
                    dtype_b="NA",
                    nan_mismatch=0,
                    max_abs_diff=np.nan,
                    mean_abs_diff=np.nan,
                    rmse=np.nan,
                    max_rel_diff=np.nan,
                    allclose=False,
                )
            )
    return results


def print_summary(df: pd.DataFrame, quiet: bool = False) -> None:
    n_total = len(df)
    n_bad = int(np.sum(df["status"] != "OK"))
    n_files = df["file_name"].nunique()
    print(f"Compared files: {n_files}")
    print(f"Compared entries: {n_total}")
    print(f"Non-OK entries: {n_bad}")

    if n_bad == 0:
        print("Result: ALL OK")
        return

    print("Result: DIFFERENCES FOUND")
    if quiet:
        bad_counts = df[df["status"] != "OK"].groupby("status").size().sort_values(ascending=False)
        print("Non-OK breakdown:")
        for k, v in bad_counts.items():
            print(f"  {k}: {v}")
        return

    bad = df[df["status"] != "OK"].copy()
    cols = ["file_name", "var_name", "status", "max_abs_diff", "rmse", "nan_mismatch"]
    if "max_abs_diff" in bad:
        bad = bad.sort_values(by=["status", "file_name", "var_name"])
    print("\nTop non-OK entries:")
    for _, r in bad.head(40).iterrows():
        print(
            f"  {r['file_name']} | {r['var_name']} | {r['status']} | "
            f"max_abs={r['max_abs_diff']} rmse={r['rmse']} nan_mismatch={int(r['nan_mismatch'])}"
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    files_a_raw = discover_files(args.dir_a, args.pattern)
    files_b_raw = discover_files(args.dir_b, args.pattern)
    files_a = build_compare_maps(files_a_raw, strict_filename=args.strict_filename)
    files_b = build_compare_maps(files_b_raw, strict_filename=args.strict_filename)

    if not files_a:
        print(f"No files in dir-a with pattern: {args.pattern}", file=sys.stderr)
        return 2
    if not files_b:
        print(f"No files in dir-b with pattern: {args.pattern}", file=sys.stderr)
        return 2

    set_a = set(files_a)
    set_b = set(files_b)
    common = sorted(set_a & set_b)
    only_a = sorted(set_a - set_b)
    only_b = sorted(set_b - set_a)

    print(f"Files in A: {len(files_a_raw)} (compare keys: {len(set_a)})")
    print(f"Files in B: {len(files_b_raw)} (compare keys: {len(set_b)})")
    print(f"Common files: {len(common)}")
    if only_a:
        print(f"Only in A ({len(only_a)}): {only_a[:10]}{' ...' if len(only_a) > 10 else ''}")
    if only_b:
        print(f"Only in B ({len(only_b)}): {only_b[:10]}{' ...' if len(only_b) > 10 else ''}")

    if not common:
        print("No common files to compare.", file=sys.stderr)
        return 2

    if args.max_files is not None:
        common = common[: int(args.max_files)]

    all_results: List[VarResult] = []
    for i, fname in enumerate(common, start=1):
        if not args.quiet:
            print(f"[{i}/{len(common)}] Comparing {fname}")
        try:
            all_results.extend(
                compare_file(
                    files_a[fname],
                    files_b[fname],
                    fname,
                    args.vars,
                    args.atol,
                    args.rtol,
                )
            )
        except Exception as e:
            all_results.append(
                VarResult(
                    file_name=fname,
                    var_name="<file>",
                    status=f"ERROR: {type(e).__name__}",
                    shape_a="NA",
                    shape_b="NA",
                    dtype_a="NA",
                    dtype_b="NA",
                    nan_mismatch=0,
                    max_abs_diff=np.nan,
                    mean_abs_diff=np.nan,
                    rmse=np.nan,
                    max_rel_diff=np.nan,
                    allclose=False,
                )
            )
            if not args.quiet:
                print(f"  ERROR: {e}")

    df = pd.DataFrame([r.__dict__ for r in all_results])
    print_summary(df, quiet=args.quiet)

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        df.to_csv(args.report, index=False)
        print(f"Saved report: {args.report}")

    return 0 if np.all(df["status"] == "OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
