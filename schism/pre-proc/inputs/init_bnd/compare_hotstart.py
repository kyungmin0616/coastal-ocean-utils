#!/usr/bin/env python3
"""
Compare hotstart.nc files and optionally compare against source data (CMEMS/HYCOM).

Usage:
  python compare_hotstart.py hotstart_a.nc hotstart_b.nc [hotstart_c.nc ...]

  # Compare hotstart vs source data
  python compare_hotstart.py hotstart.nc --src_dir /path/to/CMEMS \
      --grid ../../../grid/02/hgrid.gr3 --vgrid ../../../grid/02/vgrid.in \
      --start 2022-01-02 --source cmems --mode interp

Interpolation modes for source comparison:
  --mode interp   Trilinear interpolation (default).
  --mode nearest  Nearest neighbor in each axis (lon/lat/depth).
  --mode closest  Closest horizontal grid point (KDTree) + nearest depth.
"""
from pylib import *
import argparse
import os
import numpy as np
import glob

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

def _interp_weights(axis_nodes, targets):
    axis_nodes = np.asarray(axis_nodes)
    targets = np.asarray(targets)
    n = axis_nodes.size
    idx = np.searchsorted(axis_nodes, targets, side='right') - 1
    idx = np.clip(idx, 0, max(n-2, 0))
    den = axis_nodes[idx+1] - axis_nodes[idx]
    w = np.zeros_like(targets, dtype=float)
    ok = den != 0
    w[ok] = (targets[ok] - axis_nodes[idx[ok]]) / den[ok]
    if n > 0:
        w = np.where(targets <= axis_nodes[0], 0.0, w)
        w = np.where(targets >= axis_nodes[-1], 1.0, w)
    return idx, w

def _nearest_indices(axis_nodes, targets):
    axis_nodes = np.asarray(axis_nodes)
    targets = np.asarray(targets)
    idx = np.searchsorted(axis_nodes, targets, side='right') - 1
    idx = np.clip(idx, 0, max(len(axis_nodes) - 2, 0))
    left = axis_nodes[idx]
    right = axis_nodes[idx + 1]
    use_right = np.abs(right - targets) < np.abs(targets - left)
    idx = np.where(use_right, idx + 1, idx)
    return idx

def _parse_time_units(nc_time_var, fallback_reftime):
    vals = np.array(nc_time_var[:], dtype=float)
    units = getattr(nc_time_var, 'units', '') or ''
    ref = getattr(nc_time_var, 'reference_time', None)
    if units.lower().startswith('seconds since'):
        scale = 86400.0
    elif units.lower().startswith('hours since'):
        scale = 24.0
    elif units.lower().startswith('days since'):
        scale = 1.0
    else:
        scale = 24.0
    if 'since' in units:
        try:
            refstr = units.split('since', 1)[1].strip()
            y = int(refstr[0:4]); m = int(refstr[5:7]); d = int(refstr[8:10])
            hh = int(refstr[11:13]) if len(refstr) >= 13 else 0
            mm = int(refstr[14:16]) if len(refstr) >= 16 else 0
            ss = int(refstr[17:19]) if len(refstr) >= 19 else 0
            reftime = datenum(y, m, d) + (hh*3600 + mm*60 + ss)/86400.0
        except Exception:
            reftime = fallback_reftime
    elif ref is not None:
        reftime = ref
    else:
        reftime = fallback_reftime
    return vals / scale + reftime

def _build_source_time_index(src_dir, reftime):
    fnames = sorted([f for f in glob.glob(os.path.join(src_dir, '*.nc'))])
    if len(fnames) == 0:
        raise SystemExit('No source *.nc files found.')
    times_all = []
    file_ids = []
    time_ids = []
    for fi, fn in enumerate(fnames):
        C = ReadNC(fn, 1)
        if 'time' not in C.variables:
            C.close()
            continue
        t = _parse_time_units(C.variables['time'], reftime)
        times_all.extend(list(t))
        file_ids.extend([fi] * len(t))
        time_ids.extend(list(range(len(t))))
        C.close()
    return fnames, np.array(times_all), np.array(file_ids), np.array(time_ids)

def _nearest_time_indices(target_times, times_all):
    idx = np.searchsorted(times_all, target_times)
    idx = np.clip(idx, 1, len(times_all)-1)
    left = times_all[idx-1]
    right = times_all[idx]
    use_right = (np.abs(right - target_times) < np.abs(target_times - left))
    out = idx.copy()
    out[~use_right] = idx[~use_right] - 1
    return out

def main():
    parser = argparse.ArgumentParser(description='Compare SCHISM hotstart.nc files')
    parser.add_argument('files', nargs='+', help='hotstart.nc files to compare (first is reference)')
    parser.add_argument('--grid', default='../../../grid/02/hgrid.gr3',
                        help='SCHISM hgrid.gr3 path for plotting')
    parser.add_argument('--vgrid', default='../../../grid/02/vgrid.in',
                        help='SCHISM vgrid.in path for source comparison')
    parser.add_argument('--bad_val', type=float, default=1e3, help='bad value threshold')
    parser.add_argument('--plot', action='store_true', help='save diff plots')
    parser.add_argument('--outdir', default='hotstart_compare', help='output directory for plots')
    parser.add_argument('--src_dir', default=None, help='source data directory (CMEMS/HYCOM)')
    parser.add_argument('--start', default=None, help='start date (YYYY-MM-DD) for hotstart time')
    parser.add_argument('--source', choices=['cmems', 'hycom'], default='cmems', help='source dataset')
    parser.add_argument('--mode', choices=['interp', 'nearest', 'closest'], default='interp',
                        help='interp/nearest/closest for source comparison')
    args = parser.parse_args()

    if len(args.files) < 2 and args.src_dir is None:
        raise SystemExit('Need at least two hotstart.nc files to compare, or provide --src_dir.')

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

    if args.src_dir:
        if args.start is None:
            raise SystemExit('Provide --start (YYYY-MM-DD) for source comparison.')

        if args.source == 'cmems':
            s_temp = 'thetao'; s_salt = 'so'; s_ssh = 'zos'
            coor = ['longitude', 'latitude', 'depth']
            reftime = datenum(1950, 1, 1)
        else:
            s_temp = 'water_temp'; s_salt = 'salinity'; s_ssh = 'surf_el'
            coor = ['lon', 'lat', 'depth']
            reftime = datenum(2000, 1, 1)

        gd = read_schism_hgrid(args.grid)
        vd = read_schism_vgrid(args.vgrid)
        lxi = gd.x; lyi = gd.y
        lzi = abs(vd.compute_zcor(gd.dp)).T

        src_files, times_all, file_ids, time_ids = _build_source_time_index(args.src_dir, reftime)
        sort_idx = np.argsort(times_all)
        times_all = times_all[sort_idx]
        file_ids = file_ids[sort_idx]
        time_ids = time_ids[sort_idx]

        start_parts = [int(x) for x in args.start.split('-')]
        StartT = datenum(start_parts[0], start_parts[1], start_parts[2])
        target_time = np.array([StartT], dtype=float)
        t_sel = _nearest_time_indices(target_time, times_all)[0]
        fi = file_ids[t_sel]
        ti = time_ids[t_sel]

        C = ReadNC(src_files[fi], 1)
        sx = np.array(C.variables[coor[0]][:])
        sy = np.array(C.variables[coor[1]][:])
        sz = np.array(C.variables[coor[2]][:])
        if sz[0] != 0:
            sz[0] = 0
        lon_wrap = sx.max() > 180
        if lon_wrap:
            sx = (sx + 180) % 360 - 180
            lonidx = np.argsort(sx)
            sx = sx[lonidx]
            lxi = (lxi + 180) % 360 - 180
        else:
            lonidx = None

        if args.mode == 'interp':
            idx, ratx = _interp_weights(sx, lxi)
            idy, raty = _interp_weights(sy, lyi)
            idz, ratz = _interp_weights(sz, lzi.ravel())
            idz = idz.reshape(lzi.shape)
            ratz = ratz.reshape(lzi.shape)
            idx2 = np.broadcast_to(idx[:, None], lzi.shape)
            idy2 = np.broadcast_to(idy[:, None], lzi.shape)
            ratx2 = np.broadcast_to(ratx[:, None], lzi.shape)
            raty2 = np.broadcast_to(raty[:, None], lzi.shape)
        elif args.mode == 'nearest':
            ix_nn = _nearest_indices(sx, lxi)
            iy_nn = _nearest_indices(sy, lyi)
            iz_nn = _nearest_indices(sz, lzi.ravel()).reshape(lzi.shape)
        else:
            from scipy.spatial import cKDTree
            sxi, syi = np.meshgrid(sx, sy)
            sxy = np.c_[sxi.ravel(), syi.ravel()]
            tree = cKDTree(sxy)
            dist, flat_idx = tree.query(np.c_[lxi, lyi], k=1)
            iy2d = flat_idx // len(sx)
            ix2d = flat_idx % len(sx)
            iz_nn = _nearest_indices(sz, lzi.ravel()).reshape(lzi.shape)

        vtemp = np.array(C.variables[s_temp][ti])
        vsalt = np.array(C.variables[s_salt][ti])
        vssh = np.array(C.variables[s_ssh][ti]) if s_ssh in C.variables else None
        if lonidx is not None:
            vtemp = vtemp[:, :, lonidx]
            vsalt = vsalt[:, :, lonidx]
            if vssh is not None:
                vssh = vssh[:, lonidx]

        def _interp3d(cv):
            if args.mode == 'interp':
                v0 = np.array([
                    cv[idz,     idy2,     idx2    ], cv[idz,     idy2,     idx2 + 1],
                    cv[idz,     idy2 + 1, idx2    ], cv[idz,     idy2 + 1, idx2 + 1],
                    cv[idz + 1, idy2,     idx2    ], cv[idz + 1, idy2,     idx2 + 1],
                    cv[idz + 1, idy2 + 1, idx2    ], cv[idz + 1, idy2 + 1, idx2 + 1],
                ])
                v1 = v0[0]*(1-ratx2) + v0[1]*ratx2
                v2 = v0[2]*(1-ratx2) + v0[3]*ratx2
                v3 = v0[4]*(1-ratx2) + v0[5]*ratx2
                v4 = v0[6]*(1-ratx2) + v0[7]*ratx2
                v5 = v1*(1-raty2) + v2*raty2
                v6 = v3*(1-raty2) + v4*raty2
                return v5*(1-ratz) + v6*ratz
            if args.mode == 'nearest':
                return cv[iz_nn, iy_nn[:, None], ix_nn[:, None]]
            return cv[iz_nn, iy2d[:, None], ix2d[:, None]]

        if vssh is not None:
            if args.mode == 'interp':
                v0 = np.array([vssh[idy, idx], vssh[idy, idx+1], vssh[idy+1, idx], vssh[idy+1, idx+1]])
                v1 = v0[0]*(1-ratx) + v0[1]*ratx
                v2 = v0[2]*(1-ratx) + v0[3]*ratx
                ssh_i = v1*(1-raty) + v2*raty
            elif args.mode == 'nearest':
                ssh_i = vssh[iy_nn, ix_nn]
            else:
                ssh_i = vssh[iy2d, ix2d]
        else:
            ssh_i = None

        temp_i = _interp3d(vtemp)
        salt_i = _interp3d(vsalt)

        print('=' * 60)
        print(f"Compare {args.files[0]} vs source {args.source} ({args.mode})")
        _print_diff('temp (tr_nd[...,0])', ref.tr_nd.val[..., 0], temp_i, args.bad_val)
        _print_diff('salt (tr_nd[...,1])', ref.tr_nd.val[..., 1], salt_i, args.bad_val)
        if ssh_i is not None:
            _print_diff('eta2', ref.eta2.val, ssh_i, args.bad_val)

if __name__ == '__main__':
    main()
