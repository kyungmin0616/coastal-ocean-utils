#!/usr/bin/env python3
"""
Compare multiple hotstart.nc files (e.g., from different gen_GM_hotstart.py versions).
Prints basic stats and optionally saves diff plots for key variables.

Usage:
  python compare_hotstart.py hotstart_a.nc hotstart_b.nc [hotstart_c.nc ...]

Examples:
  # Compare two files and print stats
  python compare_hotstart.py hotstart_old.nc hotstart_new.nc

  # Compare and save diff plots
  python compare_hotstart.py hotstart_old.nc hotstart_new.nc --plot \
      --grid ../../../grid/02/hgrid.gr3 --outdir hs_diff_plots
"""
from pylib import *
import argparse
import os
import numpy as np

def _stats(a, bad_val=1e3):
    a = np.asarray(a)
    bad = np.isnan(a) | (np.abs(a) > bad_val)
    frac = float(np.sum(bad)) / float(a.size) if a.size else 0.0
    if np.all(bad):
        return frac, np.nan, np.nan, np.nan
    good = a[~bad]
    return frac, float(np.nanmin(good)), float(np.nanmax(good)), float(np.nanmean(good))

def _diff_stats(a, b, bad_val=1e3):
    d = np.asarray(a) - np.asarray(b)
    return _stats(d, bad_val=bad_val)

def _print_stats(label, a, bad_val):
    frac, vmin, vmax, vmean = _stats(a, bad_val)
    print(f"{label}: bad_frac={frac:.4f} min={vmin:.4g} max={vmax:.4g} mean={vmean:.4g}")

def _print_diff(label, a, b, bad_val):
    frac, vmin, vmax, vmean = _diff_stats(a, b, bad_val)
    print(f"{label} diff: bad_frac={frac:.4f} min={vmin:.4g} max={vmax:.4g} mean={vmean:.4g}")

def _maybe_print_stats(obj, name, bad_val):
    if not hasattr(obj, name):
        print(f"{name}: not present")
        return
    _print_stats(name, getattr(obj, name).val, bad_val)

def _maybe_print_diff(obj, ref, name, bad_val):
    if not hasattr(obj, name) or not hasattr(ref, name):
        print(f"{name} diff: not present")
        return
    _print_diff(name, getattr(obj, name).val, getattr(ref, name).val, bad_val)

def _plot_diff(gd, diff, title, outpath):
    figure(figsize=[8, 6])
    gd.plot(fmt=1, value=diff, cmap='jet')
    title and plt.title(title)
    xlabel('Longitude'); ylabel('Latitude')
    tight_layout()
    savefig(outpath, dpi=200, bbox_inches='tight')
    close()

def main():
    parser = argparse.ArgumentParser(description='Compare SCHISM hotstart.nc files')
    parser.add_argument('files', nargs='+', help='hotstart.nc files to compare (first is reference)')
    parser.add_argument('--grid', default='../../../grid/02/hgrid.gr3',
                        help='SCHISM hgrid.gr3 path for plotting')
    parser.add_argument('--bad_val', type=float, default=1e3, help='bad value threshold')
    parser.add_argument('--plot', action='store_true', help='save diff plots')
    parser.add_argument('--outdir', default='hotstart_compare', help='output directory for plots')
    args = parser.parse_args()

    if len(args.files) < 2:
        raise SystemExit('Need at least two hotstart.nc files to compare.')

    if args.plot:
        os.makedirs(args.outdir, exist_ok=True)
        gd = read_schism_hgrid(args.grid)

    ref = ReadNC(args.files[0])
    print(f"Reference: {args.files[0]}")
    _maybe_print_stats(ref, 'eta2', args.bad_val)
    if hasattr(ref, 'tr_nd'):
        _print_stats('temp (tr_nd[...,0])', ref.tr_nd.val[..., 0], args.bad_val)
        _print_stats('salt (tr_nd[...,1])', ref.tr_nd.val[..., 1], args.bad_val)
    _maybe_print_stats(ref, 'tr_el', args.bad_val)
    _maybe_print_stats(ref, 'tr_nd0', args.bad_val)
    print('-' * 60)

    for f in args.files[1:]:
        cur = ReadNC(f)
        print(f"Compare: {f}")
        _maybe_print_diff(cur, ref, 'eta2', args.bad_val)
        if hasattr(cur, 'tr_nd') and hasattr(ref, 'tr_nd'):
            _print_diff('temp (tr_nd[...,0])', cur.tr_nd.val[..., 0], ref.tr_nd.val[..., 0], args.bad_val)
            _print_diff('salt (tr_nd[...,1])', cur.tr_nd.val[..., 1], ref.tr_nd.val[..., 1], args.bad_val)
        _maybe_print_diff(cur, ref, 'tr_el', args.bad_val)
        _maybe_print_diff(cur, ref, 'tr_nd0', args.bad_val)

        if args.plot:
            _plot_diff(gd, cur.eta2.val - ref.eta2.val, 'eta2 diff', os.path.join(args.outdir, 'eta2_diff.png'))
            _plot_diff(gd, cur.tr_nd.val[:, 0, 0] - ref.tr_nd.val[:, 0, 0],
                       'temp (bottom) diff', os.path.join(args.outdir, 'temp_bottom_diff.png'))
            _plot_diff(gd, cur.tr_nd.val[:, -1, 0] - ref.tr_nd.val[:, -1, 0],
                       'temp (surface) diff', os.path.join(args.outdir, 'temp_surface_diff.png'))
            _plot_diff(gd, cur.tr_nd.val[:, 0, 1] - ref.tr_nd.val[:, 0, 1],
                       'salt (bottom) diff', os.path.join(args.outdir, 'salt_bottom_diff.png'))
            _plot_diff(gd, cur.tr_nd.val[:, -1, 1] - ref.tr_nd.val[:, -1, 1],
                       'salt (surface) diff', os.path.join(args.outdir, 'salt_surface_diff.png'))

        print('-' * 60)
        # ReadNC returns zdata; no close method needed.

if __name__ == '__main__':
    main()
