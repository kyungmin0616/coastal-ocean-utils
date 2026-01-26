#!/usr/bin/env python3
#!/usr/bin/env python3
"""
find_br_files_v3.py
==================

Purpose
-------
Quality-control utility for CMEMS subset NetCDF files named as:

    cmems_YYYY_MM_DD_HH.nc

The script performs TWO independent checks:

1) Broken / incomplete file detection
   - Size outlier detection (relative to median file size)
   - NetCDF open test
   - Dimension signature consistency vs a reference file
   - Required variable existence (optional)
   - Deep-read test on the LAST element of selected variables
     (catches truncated / partially-written files)

2) Missing data detection based on filename timestamps
   - Parses datetime from filenames (YYYY_MM_DD_HH)
   - Determines expected time sequence (daily / hourly)
   - Reports missing timestamps and missing ranges
   - Uses ONLY files that passed integrity checks ("OK files")

Outputs
-------
By default (prefix = "scan"):

  scan_report.csv
      Per-file QC status (size, dims, deep read, actions)

  scan_ok_files.txt
      List of files that passed all integrity checks

  scan_broken_files.txt
      List of files flagged as broken/incomplete

If --check-missing-times is enabled:

  scan_missing_times.txt
      One missing timestamp per line (ISO format)

  scan_missing_ranges.txt
      Missing timestamps collapsed into continuous ranges


Filename Convention Assumed
---------------------------
Files must follow:

    cmems_YYYY_MM_DD_HH.nc

Example:
    cmems_2018_09_11_00.nc  →  2018-09-11 00:00


Core Options
------------

Integrity / Broken-file checks:
  --pattern PATTERN
        Glob pattern for files (default: cmems_*.nc)

  --size-tol FLOAT
        Fractional tolerance for size outlier detection
        Default: 0.20  (±20% from median size)

  --no-size-check
        Disable size-based outlier detection

  --ref-file FILE.nc
        Reference "good" file for dimension signature comparison
        Default: median-sized file in directory

  --required-vars "v1,v2,..."
        Variables that MUST exist in every file
        Example: "zos,thetao,uo,vo"

Actions on broken files:
  --trash-dir DIR
        Move broken files to DIR (recommended)

  --delete
        Permanently delete broken files (dangerous)


Missing-time detection:
  --check-missing-times
        Enable timestamp continuity check

  --step-hours N
        Expected time step in hours
        Example: 24 for daily, 3 for 3-hourly
        If not set, inferred from filenames

  --start YYYY-MM-DDTHH
        Force expected start time
        Example: 1998-01-19T00

  --end YYYY-MM-DDTHH
        Force expected end time
        Example: 2023-10-24T00


Typical Usage Examples
----------------------

1) Dry-run QC + missing-date check (daily data, auto inference):

    python find_br_files_v3.py . \
        --pattern "cmems_*.nc" \
        --check-missing-times

2) Daily CMEMS data with explicit 24h step:

    python find_br_files_v3.py . \
        --pattern "cmems_*.nc" \
        --check-missing-times \
        --step-hours 24

3) Enforce required CMEMS variables:

    python find_br_files_v3.py . \
        --pattern "cmems_*.nc" \
        --required-vars "zos,thetao,uo,vo" \
        --check-missing-times \
        --step-hours 24

4) Move broken files safely to a trash directory:

    mkdir -p trash_broken
    python find_br_files_v3.py . \
        --pattern "cmems_*.nc" \
        --check-missing-times \
        --step-hours 24 \
        --trash-dir trash_broken

5) Strict QC over a known full period:

    python find_br_files_v3.py . \
        --pattern "cmems_*.nc" \
        --check-missing-times \
        --step-hours 24 \
        --start 1998-01-19T00 \
        --end   2023-10-24T00


Notes
-----
- Missing-time detection only counts files that PASS integrity checks.
- Broken files do NOT count as valid timestamps.
- This script is safe by default (no deletion unless --delete is used).
- Designed for HPC batch post-processing of CMEMS subsets.

"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Dict, List, Optional, Tuple

FNAME_RE = re.compile(r"^cmems_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.nc$")

def mb(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)

def parse_list(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def choose_reference_file(files: List[str]) -> str:
    sizes = [(os.stat(f).st_size, f) for f in files]
    sizes.sort()
    return sizes[len(sizes)//2][1]

def get_dims_signature(nc) -> Dict[str, int]:
    return {k: len(v) for k, v in nc.dimensions.items()}

def safe_last_read(var) -> Tuple[bool, str]:
    try:
        nd = getattr(var, "ndim", 0)
        size = getattr(var, "size", 0)
        if size == 0:
            return False, "VAR_SIZE_ZERO"
        if nd == 0:
            _ = var[()]
            return True, "READ_SCALAR_OK"
        idx = tuple(-1 for _ in range(nd))
        _ = var[idx]
        return True, "READ_LAST_OK"
    except Exception as e:
        return False, f"READ_LAST_FAIL:{type(e).__name__}:{e}"

def pick_data_vars(nc, required: List[str], max_auto: int = 3) -> List[str]:
    if required:
        return [v for v in required if v in nc.variables]

    avoid = {"time", "lon", "longitude", "lat", "latitude", "depth", "lev", "level"}
    candidates = []
    for vname, var in nc.variables.items():
        vlow = vname.lower()
        if vlow in avoid:
            continue
        try:
            if getattr(var, "ndim", 0) >= 1 and getattr(var, "size", 0) > 0:
                candidates.append(vname)
        except Exception:
            continue
    return candidates[:max_auto]

def parse_datetime_from_filename(path: str) -> Optional[datetime]:
    base = os.path.basename(path)
    m = FNAME_RE.match(base)
    if not m:
        return None
    y, mo, d, hh = map(int, m.groups())
    return datetime(y, mo, d, hh)

def infer_step(dts: List[datetime]) -> Optional[timedelta]:
    """Infer the most common step from sorted timestamps."""
    if len(dts) < 2:
        return None
    dts_sorted = sorted(dts)
    diffs = [(dts_sorted[i+1] - dts_sorted[i]) for i in range(len(dts_sorted)-1)]
    # count diffs
    counts: Dict[timedelta, int] = {}
    for df in diffs:
        counts[df] = counts.get(df, 0) + 1
    # pick the mode
    return max(counts.items(), key=lambda kv: kv[1])[0]

def generate_expected_range(start: datetime, end: datetime, step: timedelta) -> List[datetime]:
    cur = start
    out = []
    while cur <= end:
        out.append(cur)
        cur += step
    return out

def collapse_ranges(missing: List[datetime], step: timedelta) -> List[Tuple[datetime, datetime]]:
    """Collapse missing times into contiguous ranges based on step."""
    if not missing:
        return []
    missing_sorted = sorted(missing)
    ranges = []
    s = missing_sorted[0]
    prev = missing_sorted[0]
    for t in missing_sorted[1:]:
        if t - prev == step:
            prev = t
        else:
            ranges.append((s, prev))
            s = t
            prev = t
    ranges.append((s, prev))
    return ranges

@dataclass
class FileResult:
    path: str
    size_mb: float
    size_outlier: bool
    open_ok: bool
    dims_match_ref: bool
    required_vars_present: bool
    deep_read_ok: bool
    message: str
    action: str

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Directory or a single .nc file")
    ap.add_argument("--pattern", default="cmems_*.nc", help="Glob pattern (default: cmems_*.nc)")
    ap.add_argument("--report-prefix", default="scan", help="Prefix for report files (default: scan)")

    # Broken-file checks
    ap.add_argument("--size-tol", type=float, default=0.20,
                    help="Flag size outlier vs median by ±fraction (default 0.20 = 20%%).")
    ap.add_argument("--no-size-check", action="store_true", help="Disable size outlier check")
    ap.add_argument("--ref-file", default=None, help="Reference good file for dimension signature")
    ap.add_argument("--required-vars", default="", help='Comma vars required (e.g. "zos,thetao,uo,vo")')

    # Missing-time checks
    ap.add_argument("--check-missing-times", action="store_true",
                    help="Check missing timestamps based on filename datetimes.")
    ap.add_argument("--step-hours", type=int, default=None,
                    help="Force expected time step in hours (e.g. 24 for daily). If not set, inferred from data.")
    ap.add_argument("--start", default=None,
                    help='Force expected start datetime "YYYY-MM-DDTHH" (example: 2018-01-01T00)')
    ap.add_argument("--end", default=None,
                    help='Force expected end datetime "YYYY-MM-DDTHH" (example: 2018-12-31T00)')

    # Actions
    ap.add_argument("--delete", action="store_true", help="Delete broken files")
    ap.add_argument("--trash-dir", default=None, help="Move broken files to this directory instead of deleting")

    args = ap.parse_args()

    # Collect files
    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, args.pattern)))
    else:
        files = [args.path]

    if not files:
        print("No files matched.")
        sys.exit(0)

    try:
        from netCDF4 import Dataset
    except Exception as e:
        print(f"Failed to import netCDF4. Error: {e}")
        sys.exit(2)

    required = parse_list(args.required_vars)

    # median size
    sizes_bytes = []
    for f in files:
        try:
            sizes_bytes.append(os.stat(f).st_size)
        except Exception:
            sizes_bytes.append(0)
    med = median(sizes_bytes)
    med_mb = mb(int(med)) if med else 0.0

    # reference file
    ref = args.ref_file or choose_reference_file(files)
    if args.trash_dir:
        os.makedirs(args.trash_dir, exist_ok=True)

    # load ref dims
    try:
        with Dataset(ref, "r") as nc:
            ref_dims = get_dims_signature(nc)
    except Exception as e:
        print(f"Reference file cannot be opened: {ref}\n{type(e).__name__}: {e}")
        sys.exit(3)

    results: List[FileResult] = []
    broken_paths: List[str] = []
    ok_paths: List[str] = []

    report_csv = f"{args.report_prefix}_report.csv"
    broken_txt = f"{args.report_prefix}_broken_files.txt"
    ok_txt = f"{args.report_prefix}_ok_files.txt"

    with open(report_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "path", "size_mb", "median_mb", "size_outlier",
            "open_ok", "dims_match_ref", "required_vars_present",
            "deep_read_ok", "message", "action"
        ])

        for f in files:
            action = "none"
            msg_parts = []

            try:
                st = os.stat(f)
                size_mb = mb(st.st_size)
            except Exception as e:
                r = FileResult(f, float("nan"), True, False, False, False, False, f"STAT_FAIL:{e}", "none")
                results.append(r)
                broken_paths.append(f)
                w.writerow([r.path, "", f"{med_mb:.2f}", True, False, False, False, False, r.message, r.action])
                continue

            size_outlier = False
            if (not args.no_size_check) and med > 0:
                lo = med * (1.0 - args.size_tol)
                hi = med * (1.0 + args.size_tol)
                size_outlier = (st.st_size < lo) or (st.st_size > hi)
                if size_outlier:
                    msg_parts.append("SIZE_OUTLIER")

            open_ok = True
            dims_match = True
            required_present = True
            deep_read_ok = True

            try:
                with Dataset(f, "r") as nc:
                    dims = get_dims_signature(nc)
                    if dims != ref_dims:
                        dims_match = False
                        msg_parts.append("DIMS_MISMATCH")

                    if required:
                        missing = [v for v in required if v not in nc.variables]
                        if missing:
                            required_present = False
                            msg_parts.append(f"VARS_MISSING:{'|'.join(missing)}")

                    test_vars = pick_data_vars(nc, required=required, max_auto=3)
                    if not test_vars:
                        deep_read_ok = False
                        msg_parts.append("NO_TEST_VARS")
                    else:
                        for vname in test_vars:
                            ok_read, m = safe_last_read(nc.variables[vname])
                            if not ok_read:
                                deep_read_ok = False
                                msg_parts.append(f"{vname}:{m}")
                                break

            except Exception as e:
                open_ok = False
                dims_match = False
                required_present = False
                deep_read_ok = False
                msg_parts.append(f"OPEN_FAIL:{type(e).__name__}:{e}")

            is_broken = (not open_ok) or (not dims_match) or (not required_present) or (not deep_read_ok) or size_outlier

            if is_broken:
                broken_paths.append(f)
                if args.trash_dir:
                    dest = os.path.join(args.trash_dir, os.path.basename(f))
                    try:
                        os.replace(f, dest)
                        action = f"moved_to:{dest}"
                    except Exception as e:
                        action = f"move_failed:{e}"
                elif args.delete:
                    try:
                        os.remove(f)
                        action = "deleted"
                    except Exception as e:
                        action = f"delete_failed:{e}"
            else:
                ok_paths.append(f)

            message = ";".join(msg_parts) if msg_parts else "OK"
            r = FileResult(f, size_mb, size_outlier, open_ok, dims_match, required_present, deep_read_ok, message, action)
            results.append(r)

            w.writerow([
                r.path, f"{r.size_mb:.2f}", f"{med_mb:.2f}", r.size_outlier,
                r.open_ok, r.dims_match_ref, r.required_vars_present,
                r.deep_read_ok, r.message, r.action
            ])

    with open(broken_txt, "w") as fb:
        fb.write("\n".join(broken_paths) + ("\n" if broken_paths else ""))
    with open(ok_txt, "w") as fo:
        fo.write("\n".join(ok_paths) + ("\n" if ok_paths else ""))

    print(f"Scanned: {len(files)} files")
    print(f"Median size: {med_mb:.2f} MB")
    print(f"OK:      {len(ok_paths)}")
    print(f"Broken:  {len(broken_paths)}")
    print(f"Reports: {report_csv}, {broken_txt}, {ok_txt}")

    # -------------------------
    # Missing-time detection
    # -------------------------
    if args.check_missing_times:
        # Use OK files only (if a file is broken, it doesn't count as valid data)
        ok_dt = []
        ok_dt_to_path = {}
        for p in ok_paths:
            dt = parse_datetime_from_filename(p)
            if dt is not None:
                ok_dt.append(dt)
                ok_dt_to_path[dt] = p

        missing_txt = f"{args.report_prefix}_missing_times.txt"
        missing_ranges_txt = f"{args.report_prefix}_missing_ranges.txt"

        if not ok_dt:
            print("Missing-time check: no parsable cmems_YYYY_MM_DD_HH.nc filenames among OK files.")
            with open(missing_txt, "w") as f:
                f.write("")
            with open(missing_ranges_txt, "w") as f:
                f.write("")
            return

        ok_dt_sorted = sorted(set(ok_dt))

        # Determine expected range
        if args.start:
            start = datetime.strptime(args.start, "%Y-%m-%dT%H")
        else:
            start = ok_dt_sorted[0]

        if args.end:
            end = datetime.strptime(args.end, "%Y-%m-%dT%H")
        else:
            end = ok_dt_sorted[-1]

        # Determine step
        if args.step_hours is not None:
            step = timedelta(hours=args.step_hours)
        else:
            step = infer_step(ok_dt_sorted)
            if step is None:
                step = timedelta(hours=24)  # fallback

        expected = generate_expected_range(start, end, step)
        have = set(ok_dt_sorted)
        missing = [t for t in expected if t not in have]

        with open(missing_txt, "w") as f:
            for t in missing:
                f.write(t.strftime("%Y-%m-%dT%H") + "\n")

        ranges = collapse_ranges(missing, step)
        with open(missing_ranges_txt, "w") as f:
            for a, b in ranges:
                if a == b:
                    f.write(a.strftime("%Y-%m-%dT%H") + "\n")
                else:
                    f.write(f"{a.strftime('%Y-%m-%dT%H')}  ->  {b.strftime('%Y-%m-%dT%H')}\n")

        print("\nMissing-time check (based on OK files & filename timestamps):")
        print(f"  Expected start: {start.strftime('%Y-%m-%dT%H')}")
        print(f"  Expected end  : {end.strftime('%Y-%m-%dT%H')}")
        print(f"  Step          : {step}")
        print(f"  Missing count : {len(missing)}")
        print(f"  Missing lists : {missing_txt}, {missing_ranges_txt}")

if __name__ == "__main__":
    main()
