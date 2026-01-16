#!/usr/bin/env python3
"""
Generate SCHISM open-boundary *.th.nc and tracer nudging *_nu.nc from GM/CMEMS-like
products — now with optional MPI parallelism (files distributed across ranks).

Highlights:
- USER_CFG block + CLI overrides for reproducible runs
- Time-units aware parsing (seconds/hours/days since <ref>)
- Safe 2-D/3-D interpolation weights helper (prevents OOB indices)
- Great-circle nudging distances in meters (rlmax_m)
- Nudge mask logic fixed (regen if either *_nudge.gr3 missing)
- Robust NaN-repair ordering (ifix)
- Optional low-pass filter hooks
- MPI mode: scatter files to ranks; rank 0 gathers, sorts, time-interps, writes NetCDFs
"""

from pylib import *  # zdata, ReadNC, WriteNC, read_schism_hgrid, read_schism_vgrid, loadz, fexist, datenum, lpfilt, near_pts
import os
import glob
import argparse
import logging
import numpy as np
import time
from numpy import array, arange, argsort, meshgrid, c_, nonzero
from scipy import interpolate
import builtins as _bi  # ensure scalar-safe max/min

# --------------------------- MPI setup ---------------------------
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    class _Dummy:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def bcast(self, x, root=0): return x
        def barrier(self): pass
        def gather(self, x, root=0): return [x]
    COMM = _Dummy()
    RANK = 0
    SIZE = 1

# ---------------------------------------------------------------------
# USER CONFIG (edit here) + CLI overrides
# ---------------------------------------------------------------------
USER_CFG = {
    'grd': './',
    'dir_data': '/scratch3/projects/CATUFS/KyungminPark/dataset/init_bnd/CMEMS/part1/',
    'start': '2021-06-01',
    'end':   '2021-10-5',
    'dt': 1.0,                 # days
    'ibnds': [1,],              # 1-based boundary IDs
    'ifix': 1,                 # 0: fix parents first, 1: repair after gather
    'rlmax_m': 20000.0,        # nudging radius (meters)
    'rnu_day': 0.25,           # nudging timescale (days)
    # dataset vars (adjust as needed)
    'svars':  ['zos', 'thetao', 'so', ['uo','vo'], 'thetao', 'so'],
    'snames': ['elev2D.th.nc', 'TEM_3D.th.nc', 'SAL_3D.th.nc', 'uv3D.th.nc', 'TEM_nu.nc', 'SAL_nu.nc'],
    'mvars':  ['elev',         'temp',         'salt',        ['u','v'],      'temp',     'salt'],
    'iflags': [1, 1, 1, 1, 1, 1],
    'iLP': [0, 0, 0, 0, 0, 0], # optional low-pass per stream
    'fc': 0.25,                # day cutoff for lpfilt
    'coor': ['longitude', 'latitude', 'depth'],
    'log': 'INFO',
    'bad_val': 1e3,
    'io_group': 0,             # limit concurrent readers: 0=off, N=phases
    'qc': False,               # log basic QC stats per variable
}

# CLI (flags override USER_CFG)
parser = argparse.ArgumentParser(description='Generate SCHISM boundary and nudge from GM/CMEMS grids (MPI-enabled)')
parser.add_argument('--grd', default=None)
parser.add_argument('--data', dest='dir_data', default=None)
parser.add_argument('--start', default=None)
parser.add_argument('--end',   default=None)
parser.add_argument('--dt', type=float, default=None)
parser.add_argument('--ibnds', type=int, nargs='+', default=None)
parser.add_argument('--ifix', type=int, default=None, choices=[0,1])
parser.add_argument('--rlmax_m', type=float, default=None)
parser.add_argument('--rnu_day', type=float, default=None)
parser.add_argument('--log', default=None)
parser.add_argument('--bad_val', type=float, default=None)
parser.add_argument('--io_group', type=int, default=None)
parser.add_argument('--qc', action='store_true', default=None)
args = parser.parse_args()

from types import SimpleNamespace
_cfg = USER_CFG.copy()
for k, v in vars(args).items():
    if v is not None:
        _cfg[k] = v
cfg = SimpleNamespace(**_cfg)

# Rank-aware logging
if RANK == 0:
    logging.basicConfig(level=getattr(logging, cfg.log.upper(), logging.INFO),
                        format='[%(levelname)s] %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

# Broadcast cfg to all ranks
cfg = SimpleNamespace(**COMM.bcast(dict(cfg.__dict__), root=0))

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _interp_weights(axis_nodes, targets):
    """Return (idx, w) for linear interpolation along 1D axis_nodes (increasing).
    Works for scalar or N-D 'targets' arrays; shapes are preserved.
    """
    axis_nodes = np.asarray(axis_nodes)
    targets = np.asarray(targets)
    n = axis_nodes.size
    if n < 2:
        raise ValueError('axis_nodes must have length >= 2')
    idx = np.searchsorted(axis_nodes, targets, side='right') - 1
    idx = np.clip(idx, 0, max(n - 2, 0))
    den = axis_nodes[idx + 1] - axis_nodes[idx]
    w = np.zeros_like(targets, dtype=float)
    ok = den != 0
    w[ok] = (targets[ok] - axis_nodes[idx[ok]]) / den[ok]
    w = np.clip(w, 0.0, 1.0)
    w = np.where(targets <= axis_nodes[0], 0.0, w)
    w = np.where(targets >= axis_nodes[-1], 1.0, w)
    return idx, w

def _parse_time_units(nc_time_var):
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
            reftime = datenum(1950, 1, 1)
    elif ref is not None:
        reftime = ref
    else:
        reftime = datenum(1950, 1, 1)
    ctime = vals / scale + reftime
    return ctime, reftime

def _ensure_increasing(arr, name):
    arr = np.asarray(arr)
    if arr.size < 2:
        return arr, False
    if arr[1] >= arr[0]:
        return arr, False
    if RANK == 0:
        logging.info(f"{name} descending → reversing to increasing order")
    return arr[::-1].copy(), True

def _haversine_matrix(x_deg, y_deg, bx_deg, by_deg):
    """Vectorized great-circle distances (m) from all (x_deg,y_deg) to (bx_deg,by_deg)."""
    R = 6371000.0
    to_r = np.pi/180.0
    x1 = x_deg[:, None] * to_r
    y1 = y_deg[:, None] * to_r
    x2 = bx_deg[None, :] * to_r
    y2 = by_deg[None, :] * to_r
    dlat = y2 - y1
    dlon = x2 - x1
    a = np.sin(dlat/2)**2 + np.cos(y1)*np.cos(y2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def _repair_2d_nearest(values, xy, bad_val):
    """Fill bad values using nearest neighbor in 2D plane."""
    v = values.copy()
    bad = np.abs(v) > bad_val
    if not np.any(bad):
        return v
    good = ~bad
    if not np.any(good):
        return v
    v[bad] = interpolate.griddata(xy[good, :], v[good], xy[bad, :], method='nearest')
    return v

def _repair_3d_levels(values3d, xy, bad_val):
    """Repair each vertical level independently."""
    v = values3d.copy()
    for k in range(v.shape[0]):
        v[k] = _repair_2d_nearest(v[k], xy, bad_val)
    return v

def _qc_stats(arr, bad_val):
    """Return (nan_or_bad_frac, vmin, vmax) for logging."""
    a = np.asarray(arr)
    bad = np.isnan(a) | (np.abs(a) > bad_val)
    frac = float(np.sum(bad)) / float(a.size) if a.size else 0.0
    if np.all(bad):
        return frac, np.nan, np.nan
    return frac, float(np.nanmin(a[~bad])), float(np.nanmax(a[~bad]))

def _get_zcor_safe(gd, vd):
    """Return zcor[nNode, nvrt] robustly across pylib variants."""
    try:
        return compute_zcor(gd, vd)
    except Exception:
        pass
    try:
        return compute_zcor(vd, gd)
    except Exception:
        pass
    for name in ('compute_zcor', 'get_zcor', 'zcor_from_dp'):
        if hasattr(vd, name):
            fn = getattr(vd, name)
            for arg in (gd, getattr(gd, 'dp', None)):
                if arg is None: 
                    continue
                try:
                    return fn(arg)
                except Exception:
                    continue
    # Fallback: linear sigma
    nvrt = getattr(vd, 'nvrt', 20)
    dp = np.asarray(getattr(gd, 'dp'), float)
    sigma = np.linspace(-1.0, 0.0, nvrt)
    zcor = np.empty((dp.size, nvrt), dtype=float)
    for k, s in enumerate(sigma):
        zcor[:, k] = s * dp
    if RANK == 0:
        logging.warning('Using fallback linear-sigma zcor; verify vertical grid consistency.')
    return zcor

# ---------------------------------------------------------------------
# Resolve time window & vars
# ---------------------------------------------------------------------
run_t0 = time.time()
StartT = datenum(int(cfg.start[0:4]), int(cfg.start[5:7]), int(cfg.start[8:10]))
EndT   = datenum(int(cfg.end[0:4]),   int(cfg.end[5:7]),   int(cfg.end[8:10]))
iflags = cfg.iflags
svars  = cfg.svars
snames = cfg.snames
mvars  = cfg.mvars
dts    = [cfg.dt]*len(svars)
iLP    = cfg.iLP
fc     = cfg.fc
coor   = cfg.coor

# ---------------------------------------------------------------------
# Discover input files
# ---------------------------------------------------------------------
dir_data = cfg.dir_data
if os.path.isfile(f'{dir_data}/dates.out'):
    mti = array(loadtxt(f'{dir_data}/dates.out'))
    fid = open(f'{dir_data}/files.out').read().split()
    fnames = np.array(fid, dtype=object)
else:
    if RANK == 0:
        logging.warning('dates.out/files.out not found; scanning *.nc to infer chronology (slower).')
        fnames = []
        ftimes = []
        for f in sorted(glob.glob(os.path.join(dir_data, '**', '*.nc'), recursive=True)):
            try:
                C = ReadNC(f, 1)
                ctime, _ = _parse_time_units(C.variables['time'])
                ftimes.append(ctime)
                fnames.append(os.path.basename(f))
                C.close()
            except Exception as e:
                logging.error(f'Skip {f}: {e}')
        mti = np.array([np.min(t) for t in ftimes]) if ftimes else np.array([])
        fnames = np.array(fnames, dtype=object)
    mti    = COMM.bcast(mti if RANK==0 else None, root=0)
    fnames = COMM.bcast(fnames if RANK==0 else None, root=0)

if fnames.size == 0:
    if RANK == 0:
        raise RuntimeError('No input files discovered.')
    raise SystemExit

# Filter by time window (±1 day buffer)
fpt = (mti >= (StartT - 1)) * (mti < (EndT + 1))
fnames = fnames[fpt]
mti    = mti[fpt]
sind   = argsort(mti)
mti    = mti[sind]
fnames = fnames[sind]

if RANK == 0:
    logging.info(f'Found {len(fnames)} source files spanning {len(mti)} time anchors.')

# ---------------------------------------------------------------------
# Read SCHISM grid on root; broadcast essentials
# ---------------------------------------------------------------------
if RANK == 0:
    grddir = cfg.grd
    if fexist(grddir + '/grid.npz'):
        gd = loadz(grddir + '/grid.npz').hgrid
        vd = loadz(grddir + '/grid.npz').vgrid
        gd.x, gd.y = gd.lon, gd.lat
    else:
        gd = read_schism_hgrid(grddir + '/hgrid.ll')
        vd = read_schism_vgrid(grddir + '/vgrid.in')

    # Prepare nudging masks (if missing)
    need_nudge = (not os.path.isfile('TEM_nudge.gr3')) or (not os.path.isfile('SAL_nudge.gr3'))
    if need_nudge:
        logging.info('Generating TEM_nudge.gr3 and SAL_nudge.gr3 using haversine (meters)')
        nudge_coeff = np.zeros(len(gd.x), dtype=float)
        rnu_max = 1.0 / cfg.rnu_day / 86400.0
        for ibnd in cfg.ibnds:
            bnd_idx = gd.iobn[ibnd - 1]
            dis = _haversine_matrix(gd.x, gd.y, gd.x[bnd_idx], gd.y[bnd_idx]).min(axis=1)
            tmp = (1.0 - dis / cfg.rlmax_m) * rnu_max
            tmp[tmp < 0] = 0
            tmp[tmp > rnu_max] = rnu_max
            fp = tmp > 0
            nudge_coeff[fp] = tmp[fp]
        gd.write_hgrid('./TEM_nudge.gr3', value=nudge_coeff)
        gd.write_hgrid('./SAL_nudge.gr3', value=nudge_coeff)

    nvrt = vd.nvrt
    # Build boundary node list
    bind = gd.iobn[cfg.ibnds[0] - 1]
    for ib in cfg.ibnds[1:]:
        bind = np.r_[bind, gd.iobn[ib - 1]]
    nobn = len(bind)
    lxi = gd.x[bind]; lyi = gd.y[bind]
    # Vertical target depths (match gen_GM_3Dth_nudge.py)
    if vd.ivcor == 2:
        lzi = abs(compute_zcor(vd.sigma, gd.dp[bind], ivcor=2, vd=vd))
    else:
        lzi = abs(compute_zcor(vd.sigma[bind], gd.dp[bind]))

    root_payload = {
        'bind': bind, 'nobn': nobn, 'lxi': lxi, 'lyi': lyi, 'lzi': lzi, 'nvrt': nvrt,
    }
else:
    root_payload = None

root_payload = COMM.bcast(root_payload, root=0)
bind = root_payload['bind']
nobn = root_payload['nobn']
lxi  = root_payload['lxi']
lyi  = root_payload['lyi']
lzi  = root_payload['lzi']
nvrt = root_payload['nvrt']

# ---------------------------------------------------------------------
# Work per stream
# ---------------------------------------------------------------------
for n, (sname, svar, mvar, dt, iflag) in enumerate(zip(snames, svars, mvars, dts, iflags)):
    if isinstance(svar, str):
        svar = [svar]
        mvar = [mvar]
    if iflag == 0:
        if RANK == 0:
            logging.info(f'Skipping {sname} (iflag=0)')
        continue

    if RANK == 0:
        logging.info(f'Building {sname} (MPI ranks={SIZE})')
    logging.info(f'Rank {RANK}: start {sname} nobn={nobn} nvrt={nvrt}')

    local_time = []
    local_series = {mv: [] for mv in mvar}

    # Rank-local caches for interp weights
    sx0 = sy0 = sz0 = None
    idx0 = idy0 = ratx0 = raty0 = None
    idx = idy = idz = None
    ratx = raty = ratz = None
    sxy = bxy = None
    bxyz_flat = None

    # Distribute files across ranks
    my_files = fnames[RANK::SIZE]

    phases = cfg.io_group if (cfg.io_group and cfg.io_group < SIZE) else 1
    for phase in range(phases):
        if (RANK % phases) != phase:
            COMM.barrier()
            continue
        for fname in my_files:
            logging.info(f'Rank {RANK}: file {fname} for {sname}')
            fp = fname
            try:
                C = ReadNC(fp, 1)
            except Exception as e:
                if RANK == 0:
                    logging.error(f'Rank {RANK}: failed to open {fname}: {e}')
                continue

            ctime, _ = _parse_time_units(C.variables['time'])
            local_time.extend(ctime)

            # Coords
            sx = array(C.variables[coor[0]][:])
            sy = array(C.variables[coor[1]][:])
            sz = array(C.variables[coor[2]][:])
            nz = len(sz)

            # Ensure lzi within source-depth range
            lzi_clamped = lzi.copy()
            fpz = lzi_clamped >= sz.max()
            lzi_clamped[fpz] = sz.max() - 1e-6

            # Longitude wrap to [-180,180]
            lonidx = None
            if sx.max() > 180:
                sx = (sx + 180) % 360 - 180
                lonidx = argsort(sx)
                sx = sx[lonidx]
            # Latitude increasing
            sy, flipped_lat = _ensure_increasing(sy, 'latitude')

            # Rebuild weights / repair maps if grid changed
            if (sx0 is None) or (not np.array_equal(sx, sx0)) or (not np.array_equal(sy, sy0)) or (not np.array_equal(sz, sz0)):
                sxi, syi = meshgrid(sx, sy)
                sxy = c_[sxi.ravel(), syi.ravel()]
                bxy = c_[lxi, lyi]
                bxyz_flat = c_[np.repeat(lxi, nvrt), np.repeat(lyi, nvrt), lzi_clamped.ravel()]

                sindns = sindps = None
                if len(svar) > 1:
                    sample_var = svar[1]
                    cvs = array(C.variables[sample_var][0])
                    if lonidx is not None:
                        cvs = cvs[:, :, lonidx]
                    if flipped_lat:
                        cvs = cvs[:, ::-1, :]
                    sindns, sindps = [], []
                    for ii in arange(nz):
                        cv = cvs[ii]
                        r = cv.ravel()
                        fpn = np.abs(r) > cfg.bad_val
                        sindn = nonzero(fpn)[0]
                        sindr = nonzero(~fpn)[0]
                        if len(sindr) != 0:
                            sindp = sindr[near_pts(sxy[sindn], sxy[sindr])]
                        else:
                            sindp = array([])
                        sindns.append(sindn); sindps.append(sindp)

                sx0 = sx.copy(); sy0 = sy.copy(); sz0 = sz.copy()
                # 2D weights for elevation
                idx0, ratx0 = _interp_weights(sx0, lxi)
                idy0, raty0 = _interp_weights(sy0, lyi)
                # 3D weights for fields
                idx, ratx = _interp_weights(sx0, lxi)
                idy, raty = _interp_weights(sy0, lyi)
                idz, ratz = _interp_weights(sz0, lzi_clamped)
                # Broadcast XY to (nobn, nvrt) to match idz shape
                idx2  = np.broadcast_to(idx[:,  None], lzi_clamped.shape)
                idy2  = np.broadcast_to(idy[:,  None], lzi_clamped.shape)
                ratx2 = np.broadcast_to(ratx[:, None], lzi_clamped.shape)
                raty2 = np.broadcast_to(raty[:, None], lzi_clamped.shape)

            # Iterate times
            for i, cti in enumerate(ctime):
                for svari, mvari in zip(svar, mvar):
                    if i == 0 or i % 10 == 0:
                        logging.info(f'Rank {RANK}: {sname} var={mvari} time_idx={i} nobn={nobn} nvrt={nvrt}')
                    cv = array(C.variables[svari][i])
                    # Orient lon/lat
                    if lonidx is not None:
                        if mvari == 'elev':
                            cv = cv[:, lonidx]
                        else:
                            cv = cv[:, :, lonidx]
                    if flipped_lat:
                        if mvari == 'elev':
                            cv = cv[::-1, :]
                        else:
                            cv = cv[:, ::-1, :]

                    # Entire-field missing → fallback to last
                    if np.sum(np.abs(cv) < cfg.bad_val) == 0:
                        last = local_series[mvari][-1] if len(local_series[mvari]) else (np.zeros((nobn,)) if mvari == 'elev' else np.zeros((nobn, nvrt)))
                        local_series[mvari].append(last)
                        continue

                    if mvari == 'elev':  # 2D bilinear
                        if cfg.ifix == 0 and sxy is not None:
                            cv = _repair_2d_nearest(cv, sxy, cfg.bad_val)

                        v0 = np.array([
                            cv[idy0,     idx0    ], cv[idy0,     idx0 + 1],
                            cv[idy0 + 1, idx0    ], cv[idy0 + 1, idx0 + 1]
                        ])
                        if cfg.ifix == 1:
                            for ii in range(4):
                                v0[ii] = _repair_2d_nearest(v0[ii], bxy, cfg.bad_val)
                        v1 = v0[0] * (1 - ratx0) + v0[1] * ratx0
                        v2 = v0[2] * (1 - ratx0) + v0[3] * ratx0
                        vi = v1 * (1 - raty0) + v2 * raty0

                    else:  # 3D tri-linear
                        if cfg.ifix == 0 and isinstance(sindns, list):
                            for ii in arange(nz):
                                # level fully missing → copy from above if possible
                                if np.sum(np.abs(cv[ii]) < cfg.bad_val) == 0 and ii > 0:
                                    cv[ii] = cv[ii - 1]
                                sindn, sindp = sindns[ii], sindps[ii]
                                if len(sindp) != 0:
                                    cvi = cv[ii].ravel()
                                    cvi[sindn] = cvi[sindp]
                                    bad = np.abs(cvi) > cfg.bad_val
                                    if np.any(bad):
                                        good = ~bad
                                        if np.any(good):
                                            cvi[bad] = interpolate.griddata(sxy[good, :], cvi[good], sxy[bad, :], method='nearest')
                                    cv[ii] = cvi.reshape(cv[ii].shape)

                        v0 = np.array([
                            cv[idz,     idy2,     idx2    ], cv[idz,     idy2,     idx2 + 1],
                            cv[idz,     idy2 + 1, idx2    ], cv[idz,     idy2 + 1, idx2 + 1],
                            cv[idz + 1, idy2,     idx2    ], cv[idz + 1, idy2,     idx2 + 1],
                            cv[idz + 1, idy2 + 1, idx2    ], cv[idz + 1, idy2 + 1, idx2 + 1]
                        ])
                        if cfg.ifix == 1:
                            # match gen_GM_3Dth_nudge.py: repair in 3D xyz space
                            for ii in range(8):
                                parent = v0[ii].ravel()
                                fpn = np.abs(parent) > cfg.bad_val
                                if np.any(fpn) and np.any(~fpn):
                                    parent[fpn] = interpolate.griddata(
                                        bxyz_flat[~fpn, :], parent[~fpn], bxyz_flat[fpn, :],
                                        method='nearest', rescale=True
                                    )
                                v0[ii] = parent.reshape(v0[ii].shape)
                        v1 = v0[0] * (1 - ratx2) + v0[1] * ratx2
                        v2 = v0[2] * (1 - ratx2) + v0[3] * ratx2
                        v3 = v0[4] * (1 - ratx2) + v0[5] * ratx2
                        v4 = v0[6] * (1 - ratx2) + v0[7] * ratx2
                        v5 = v1 * (1 - raty2) + v2 * raty2
                        v6 = v3 * (1 - raty2) + v4 * raty2
                        vi = v5 * (1 - ratz) + v6 * ratz

                    local_series[mvari].append(vi)

            C.close()
        COMM.barrier()

    # -------------------- Gather to root --------------------
    COMM.barrier()
    all_times = COMM.gather(np.array(local_time, dtype=float), root=0)
    all_series = {mv: COMM.gather(np.array(local_series[mv], dtype=object), root=0) for mv in mvar}

    if RANK != 0:
        continue

    # -------------------- Root concatenation --------------------
    if len(all_times) == 0 or all_times[0].size == 0:
        logging.warning(f'No data gathered for {sname}; skipping')
        continue

    S_time = np.concatenate(all_times)
    sdict = {'time': S_time}
    for mv in mvar:
        parts = [arr for chunk in all_series[mv] for arr in chunk]
        sdict[mv] = np.stack(parts, axis=0)

    # Sort by time
    sind = np.argsort(sdict['time'])
    sdict['time'] = sdict['time'][sind]
    for mv in mvar:
        sdict[mv] = sdict[mv][sind]

    # Time interpolation → mtime (with overlap clipping)
    mtime = arange(StartT, EndT + dt + 1e-9, dt)
    tmin = float(np.min(sdict['time'])); tmax = float(np.max(sdict['time']))
    mmin = _bi.max(float(mtime[0]), tmin)
    mmax = _bi.min(float(mtime[-1]), tmax)
    if mmin > mmax:
        raise RuntimeError('No time overlap between source and requested range')
    if mmin > float(mtime[0]) or mmax < float(mtime[-1]):
        logging.warning(f'Clipping output time to overlap: [{mmin:.6f}, {mmax:.6f}] days')
    mask = (mtime >= mmin) & (mtime <= mmax)
    mtime = mtime[mask]
    nt = len(mtime)

    for mv in mvar:
        f = interpolate.interp1d(sdict['time'], sdict[mv], axis=0, bounds_error=False,
                                 fill_value=(sdict[mv][0], sdict[mv][-1]))
        svi = f(mtime)
        if iLP[n] == 1:
            svi = lpfilt(svi, dt, fc)
        sdict[mv] = svi
        if cfg.qc:
            frac, vmin, vmax = _qc_stats(svi, cfg.bad_val)
            logging.info(f'QC {sname}:{mv} bad_frac={frac:.4f} min={vmin:.4g} max={vmax:.4g}')

    # Reshape 3D variables to [nt, nobn, nvrt]
    if 'elev' in mvar:
        others = [v for v in mvar if v != 'elev']
    else:
        others = mvar[:]
    for mv in others:
        if mv in ['temp', 'salt', 'u', 'v']:
            sdict[mv] = sdict[mv].reshape([nt, nobn, nvrt])

    # -------------------- Write NetCDF (root only) --------------------
    if sname.endswith('.th.nc'):
        dimname = ['nOpenBndNodes', 'nLevels', 'nComponents', 'one', 'time']
        if sname == 'elev2D.th.nc':
            dims = [nobn, 1, 1, 1, nt]
            vi = sdict['elev'][..., None, None]
        elif sname == 'uv3D.th.nc':
            dims = [nobn, nvrt, 2, 1, nt]
            vi = np.c_[sdict['u'][..., None], sdict['v'][..., None]]
        else:  # TEM_3D / SAL_3D
            dims = [nobn, nvrt, 1, 1, nt]
            vi = sdict[mvar[0]][..., None]
        nd = zdata(); nd.dimname = dimname; nd.dims = dims
        z = zdata(); z.attrs = ['long_name']; z.long_name = 'time step (sec)'; z.dimname = ('one',);  z.val = array(dt * 86400.0); nd.time_step = z
        z = zdata(); z.attrs = ['long_name']; z.long_name = 'time (sec)';     z.dimname = ('time',); z.val = (mtime - mtime[0]) * 86400.0; nd.time = z
        z = zdata(); z.dimname = ('time', 'nOpenBndNodes', 'nLevels', 'nComponents'); z.val = vi.astype('float32'); nd.time_series = z
    else:
        dimname = ['time', 'node', 'nLevels', 'one']
        dims = [nt, nobn, nvrt, 1]
        vi = sdict[mvar[0]][..., None]
        nd = zdata(); nd.dimname = dimname; nd.dims = dims
        z = zdata(); z.dimname = ('time',); z.val = (mtime - mtime[0]) * 86400.0; nd.time = z
        z = zdata(); z.dimname = ('node',); z.val = bind + 1; nd.map_to_global_node = z
        z = zdata(); z.dimname = ('time', 'node', 'nLevels', 'one'); z.val = vi.astype('float32'); nd.tracer_concentration = z

    WriteNC(sname, nd)
    logging.info(f'Wrote {sname}')

if RANK == 0:
    logging.info('All done.')
    logging.info(f'Total runtime: {time.time()-run_t0:.2f} s')
