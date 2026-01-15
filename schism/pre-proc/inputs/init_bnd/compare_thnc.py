#!/usr/bin/env python3
"""
Compare SCHISM *.th.nc files (elev2D, uv3D, TEM_3D, SAL_3D) from different versions.

Usage:
  python compare_thnc.py file_a.th.nc file_b.th.nc [file_c.th.nc ...]
  python compare_thnc.py --ref_dir path/to/ref --cmp_dir path/to/cmp

Examples:
  python compare_thnc.py elev2D.th.nc elev2D_new.th.nc
  python compare_thnc.py uv3D.th.nc uv3D_new.th.nc
  python compare_thnc.py --ref_dir v1 --cmp_dir v2 --plot --per_bnd
"""
from pylib import *
import argparse
import numpy as np
import os
import glob

def _stats(a, bad_val=1e3):
    a = np.asarray(a, dtype=float)
    bad = np.isnan(a) | (np.abs(a) > bad_val)
    frac = float(bad.sum()) / float(a.size) if a.size else 0.0
    if bad.all():
        return frac, np.nan, np.nan, np.nan
    good = a[~bad]
    return frac, float(np.nanmin(good)), float(np.nanmax(good)), float(np.nanmean(good))

def _print_stats(label, a, bad_val):
    frac, vmin, vmax, vmean = _stats(a, bad_val)
    print(f"{label}: bad_frac={frac:.4f} min={vmin:.4g} max={vmax:.4g} mean={vmean:.4g}")

def _print_diff(label, a, b, bad_val):
    _print_stats(label + " diff", np.asarray(a) - np.asarray(b), bad_val)

def _print_time_info(nc):
    if hasattr(nc, 'time'):
        t = nc.time.val
        if hasattr(nc, 'time_step'):
            dt = nc.time_step.val
            print(f"time: n={len(t)} dt_sec={float(dt):.4g} range_sec=({float(t[0]):.4g},{float(t[-1]):.4g})")
        else:
            print(f"time: n={len(t)} range=({float(t[0]):.4g},{float(t[-1]):.4g})")
    else:
        print("time: not present")

def _print_series_stats(nc, bad_val):
    if not hasattr(nc, 'time_series'):
        print("time_series: not present")
        return
    ts = nc.time_series.val
    # Expected dims: (time, nOpenBndNodes, nLevels, nComponents)
    ncomp = ts.shape[-1] if ts.ndim >= 4 else 1
    if ncomp == 1:
        _print_stats("time_series", ts, bad_val)
    else:
        for c in range(ncomp):
            _print_stats(f"time_series comp={c}", ts[..., c], bad_val)

def _print_series_diff(nc, ref, bad_val):
    if not hasattr(nc, 'time_series') or not hasattr(ref, 'time_series'):
        print("time_series diff: not present")
        return
    a = nc.time_series.val
    b = ref.time_series.val
    ncomp = a.shape[-1] if a.ndim >= 4 else 1
    if ncomp == 1:
        _print_diff("time_series", a, b, bad_val)
    else:
        for c in range(ncomp):
            _print_diff(f"time_series comp={c}", a[..., c], b[..., c], bad_val)

def _time_days(nc):
    if hasattr(nc, 'time'):
        return array(nc.time.val) / 86400.0
    return None

def _pick_level(ts, level_idx):
    if ts.ndim < 3:
        return ts
    nlev = ts.shape[2]
    if nlev == 1:
        return ts[:, :, 0]
    return ts[:, :, level_idx]

def _plot_ts(times_day, bnd_idx, data, title, outpath):
    figure(figsize=[10, 5])
    contourf(times_day, bnd_idx, data.T, levels=60, cmap='jet')
    colorbar()
    xlabel('days')
    ylabel('Boundary node')
    title and plt.title(title)
    tight_layout()
    savefig(outpath, dpi=200, bbox_inches='tight')
    close()

def _plot_diff(nc, ref, outdir, level_idx):
    if not hasattr(nc, 'time_series') or not hasattr(ref, 'time_series'):
        return
    ts = nc.time_series.val
    rs = ref.time_series.val
    times = _time_days(nc)
    if times is None:
        return
    bnd_idx = arange(ts.shape[1])
    ncomp = ts.shape[-1] if ts.ndim >= 4 else 1

    if ncomp == 1:
        d = _pick_level(ts - rs, level_idx)
        _plot_ts(times, bnd_idx, d, 'diff comp=0', os.path.join(outdir, 'diff_comp0.png'))
    else:
        for c in range(ncomp):
            d = _pick_level(ts[..., c] - rs[..., c], level_idx)
            _plot_ts(times, bnd_idx, d, f'diff comp={c}', os.path.join(outdir, f'diff_comp{c}.png'))

def _per_bnd_stats(nc, ref, bad_val, limit):
    if not hasattr(nc, 'time_series') or not hasattr(ref, 'time_series'):
        print("per_bnd: time_series not present")
        return
    a = nc.time_series.val
    b = ref.time_series.val
    diff = a - b
    nbn = diff.shape[1]
    print(f"per_bnd: total={nbn}")
    nprint = nbn if limit <= 0 else min(nbn, limit)
    for i in range(nprint):
        d = diff[:, i, ...]
        frac, vmin, vmax, vmean = _stats(d, bad_val)
        print(f"bnd {i}: bad_frac={frac:.4f} min={vmin:.4g} max={vmax:.4g} mean={vmean:.4g}")
    if nprint < nbn:
        print(f"per_bnd: truncated to first {nprint} boundaries (use --per_bnd_limit 0 for all)")

def main():
    parser = argparse.ArgumentParser(description='Compare SCHISM *.th.nc files')
    parser.add_argument('files', nargs='*', help='*.th.nc files (first is reference)')
    parser.add_argument('--bad_val', type=float, default=1e3, help='bad value threshold')
    parser.add_argument('--plot', action='store_true', help='save diff plots')
    parser.add_argument('--outdir', default='thnc_compare', help='output directory for plots')
    parser.add_argument('--level', type=int, default=-1, help='level index for plots (default: surface)')
    parser.add_argument('--per_bnd', action='store_true', help='print per-boundary diff stats')
    parser.add_argument('--per_bnd_limit', type=int, default=50, help='max boundaries to print (0=all)')
    parser.add_argument('--ref_dir', default=None, help='reference directory with *.th.nc')
    parser.add_argument('--cmp_dir', default=None, help='comparison directory with *.th.nc')
    args = parser.parse_args()

    if args.ref_dir and args.cmp_dir:
        ref_dir = args.ref_dir
        cmp_dir = args.cmp_dir
        ref_files = {os.path.basename(f): f for f in glob.glob(os.path.join(ref_dir, '*.th.nc'))}
        cmp_files = {os.path.basename(f): f for f in glob.glob(os.path.join(cmp_dir, '*.th.nc'))}
        common = sorted(set(ref_files.keys()) & set(cmp_files.keys()))
        if len(common) == 0:
            raise SystemExit('No matching *.th.nc files found between ref_dir and cmp_dir.')
        for fname in common:
            print('=' * 60)
            print(f"Reference: {ref_files[fname]}")
            ref = ReadNC(ref_files[fname])
            _print_time_info(ref)
            _print_series_stats(ref, args.bad_val)
            print('-' * 60)
            print(f"Compare: {cmp_files[fname]}")
            cur = ReadNC(cmp_files[fname])
            _print_time_info(cur)
            _print_series_diff(cur, ref, args.bad_val)
            if args.per_bnd:
                _per_bnd_stats(cur, ref, args.bad_val, args.per_bnd_limit)
            if args.plot:
                outdir = os.path.join(args.outdir, fname.replace('.th.nc', ''))
                os.makedirs(outdir, exist_ok=True)
                _plot_diff(cur, ref, outdir, args.level)
            print('-' * 60)
        return

    if len(args.files) < 2:
        raise SystemExit('Need at least two *.th.nc files to compare, or use --ref_dir/--cmp_dir.')

    ref = ReadNC(args.files[0])
    print(f"Reference: {args.files[0]}")
    _print_time_info(ref)
    _print_series_stats(ref, args.bad_val)
    print('-' * 60)

    for f in args.files[1:]:
        cur = ReadNC(f)
        print(f"Compare: {f}")
        _print_time_info(cur)
        _print_series_diff(cur, ref, args.bad_val)
        if args.per_bnd:
            _per_bnd_stats(cur, ref, args.bad_val, args.per_bnd_limit)
        if args.plot:
            os.makedirs(args.outdir, exist_ok=True)
            _plot_diff(cur, ref, args.outdir, args.level)
        print('-' * 60)

if __name__ == '__main__':
    main()
