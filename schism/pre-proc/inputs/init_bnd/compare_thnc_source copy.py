#!/usr/bin/env python3
"""
Compare SCHISM *.th.nc files against source data (CMEMS/HYCOM).

Usage:
  python compare_thnc_source.py --thnc_dir ./ --src_dir /path/to/src \
      --grid ../../../grid/02/hgrid.gr3 --vgrid ../../../grid/02/vgrid.in \
      --ibnds 1 2 --start 2022-01-02 --source cmems

Examples:
  python compare_thnc_source.py --files elev2D.th.nc TEM_3D.th.nc \
      --src_dir /data/CMEMS/EastAsia --ibnds 1 --start 2022-01-02 --source cmems
"""
from pylib import *
import argparse
import os
import glob
import numpy as np

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
    ctime = vals / scale + reftime
    return ctime

def _stats(a, bad_val=1e3):
    a = np.asarray(a, dtype=float)
    bad = np.isnan(a) | (np.abs(a) > bad_val)
    frac = float(bad.sum()) / float(a.size) if a.size else 0.0
    if bad.all():
        return frac, np.nan, np.nan, np.nan, np.nan
    good = a[~bad]
    rmse = float(np.sqrt(np.nanmean(good * good)))
    return frac, float(np.nanmin(good)), float(np.nanmax(good)), float(np.nanmean(good)), rmse

def _print_diff_stats(label, diff, bad_val):
    frac, vmin, vmax, vmean, rmse = _stats(diff, bad_val)
    print(f"{label}: bad_frac={frac:.4f} min={vmin:.4g} max={vmax:.4g} mean={vmean:.4g} rmse={rmse:.4g}")

def _load_grid(hgrid, vgrid, ibnds):
    gd = read_schism_hgrid(hgrid)
    vd = read_schism_vgrid(vgrid)
    bind = []
    for ib in ibnds:
        bind.extend(gd.iobn[ib-1])
    bind = np.array(bind)
    lxi = gd.x[bind]
    lyi = gd.y[bind]
    if vd.ivcor == 2:
        lzi = abs(compute_zcor(vd.sigma, gd.dp[bind], ivcor=2, vd=vd))
    else:
        lzi = abs(compute_zcor(vd.sigma[bind], gd.dp[bind]))
    return gd, vd, bind, lxi, lyi, lzi

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
    parser = argparse.ArgumentParser(description='Compare *.th.nc with source data')
    parser.add_argument('--files', nargs='*', help='th.nc files to compare')
    parser.add_argument('--thnc_dir', default=None, help='directory with *.th.nc')
    parser.add_argument('--src_dir', required=True, help='source data directory')
    parser.add_argument('--grid', required=True, help='path to hgrid.gr3')
    parser.add_argument('--vgrid', required=True, help='path to vgrid.in')
    parser.add_argument('--ibnds', type=int, nargs='+', required=True, help='open boundary IDs (1-based)')
    parser.add_argument('--start', required=True, help='start date (YYYY-MM-DD) for th.nc time')
    parser.add_argument('--source', choices=['cmems','hycom'], default='cmems', help='source dataset')
    parser.add_argument('--bad_val', type=float, default=1e3, help='bad value threshold')
    args = parser.parse_args()

    if args.files:
        th_files = args.files
    elif args.thnc_dir:
        th_files = sorted(glob.glob(os.path.join(args.thnc_dir, '*.th.nc')))
    else:
        raise SystemExit('Provide --files or --thnc_dir.')

    if len(th_files) == 0:
        raise SystemExit('No *.th.nc files found.')

    if args.source == 'cmems':
        var_map = {
            'elev2D.th.nc': ('zos', '2d'),
            'TEM_3D.th.nc': ('thetao', '3d'),
            'SAL_3D.th.nc': ('so', '3d'),
            'uv3D.th.nc': (['uo', 'vo'], '3d_uv'),
        }
        coor = ['longitude', 'latitude', 'depth']
        reftime = datenum(1950, 1, 1)
    else:
        var_map = {
            'elev2D.th.nc': ('surf_el', '2d'),
            'TEM_3D.th.nc': ('water_temp', '3d'),
            'SAL_3D.th.nc': ('salinity', '3d'),
            'uv3D.th.nc': (['water_u', 'water_v'], '3d_uv'),
        }
        coor = ['lon', 'lat', 'depth']
        reftime = datenum(2000, 1, 1)

    gd, vd, bind, lxi, lyi, lzi = _load_grid(args.grid, args.vgrid, args.ibnds)
    nvrt = lzi.shape[1]

    # source time index
    src_files, times_all, file_ids, time_ids = _build_source_time_index(args.src_dir, reftime)
    sort_idx = np.argsort(times_all)
    times_all = times_all[sort_idx]
    file_ids = file_ids[sort_idx]
    time_ids = time_ids[sort_idx]

    # set up spatial weights using first source file
    C0 = ReadNC(src_files[0], 1)
    sx = np.array(C0.variables[coor[0]][:])
    sy = np.array(C0.variables[coor[1]][:])
    sz = np.array(C0.variables[coor[2]][:])
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
    idx, ratx = _interp_weights(sx, lxi)
    idy, raty = _interp_weights(sy, lyi)
    idz, ratz = _interp_weights(sz, lzi.ravel())
    idz = idz.reshape(lzi.shape)
    ratz = ratz.reshape(lzi.shape)
    idx2 = np.broadcast_to(idx[:, None], lzi.shape)
    idy2 = np.broadcast_to(idy[:, None], lzi.shape)
    ratx2 = np.broadcast_to(ratx[:, None], lzi.shape)
    raty2 = np.broadcast_to(raty[:, None], lzi.shape)
    C0.close()

    start_parts = [int(x) for x in args.start.split('-')]
    StartT = datenum(start_parts[0], start_parts[1], start_parts[2])

    for thf in th_files:
        base = os.path.basename(thf)
        if base not in var_map:
            print(f"Skip {thf}: unknown mapping for {base}")
            continue
        src_var, vtype = var_map[base]
        print('=' * 60)
        print(f"Compare {thf} vs source {args.source} ({src_var})")
        T = ReadNC(thf)
        th_time = StartT + np.array(T.time.val) / 86400.0
        ts = T.time_series.val
        t_idx = _nearest_time_indices(th_time, times_all)

        diff_all = []
        cache_file = None
        cache_nc = None

        for it, sel in enumerate(t_idx):
            fi = file_ids[sel]
            ti = time_ids[sel]
            src_file = src_files[fi]
            if cache_file != src_file:
                if cache_nc is not None:
                    cache_nc.close()
                cache_nc = ReadNC(src_file, 1)
                cache_file = src_file
            if vtype == '2d':
                cv = np.array(cache_nc.variables[src_var][ti])
                if lonidx is not None:
                    cv = cv[:, lonidx]
                v0 = np.array([cv[idy, idx], cv[idy, idx+1], cv[idy+1, idx], cv[idy+1, idx+1]])
                v1 = v0[0]*(1-ratx) + v0[1]*ratx
                v2 = v0[2]*(1-ratx) + v0[3]*ratx
                vi = v1*(1-raty) + v2*raty
                diff = ts[it, :, 0, 0] - vi
                diff_all.append(diff)
            elif vtype == '3d':
                cv = np.array(cache_nc.variables[src_var][ti])
                if lonidx is not None:
                    cv = cv[:, :, lonidx]
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
                vi = v5*(1-ratz) + v6*ratz
                diff = ts[it, :, :, 0] - vi
                diff_all.append(diff)
            elif vtype == '3d_uv':
                ctu = np.array(cache_nc.variables[src_var[0]][ti])
                ctv = np.array(cache_nc.variables[src_var[1]][ti])
                if lonidx is not None:
                    ctu = ctu[:, :, lonidx]
                    ctv = ctv[:, :, lonidx]
                def _interp3d(cv):
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
                ui = _interp3d(ctu)
                vi = _interp3d(ctv)
                diff_u = ts[it, :, :, 0] - ui
                diff_v = ts[it, :, :, 1] - vi
                diff_all.append(np.stack([diff_u, diff_v], axis=-1))
            else:
                continue

        diff_all = np.array(diff_all)
        _print_diff_stats('diff', diff_all, args.bad_val)

        if cache_nc is not None:
            cache_nc.close()

if __name__ == '__main__':
    main()
