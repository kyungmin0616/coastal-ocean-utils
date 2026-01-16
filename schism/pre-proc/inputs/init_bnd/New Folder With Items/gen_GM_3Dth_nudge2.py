#!/usr/bin/env python3
"""
Generate SCHISM open-boundary *.th.nc and tracer nudging *_nu.nc from GM/CMEMS-like
products — now with optional MPI parallelism (files distributed across ranks).

Highlights vs. baseline:
- USER_CFG block + CLI overrides for reproducible runs
- Time-units aware parsing (seconds/hours/days since <ref>)
- Safe 2-D/3-D interpolation weights helper (prevents OOB indices)
- Great-circle nudging distances in meters (rlmax_m)
- Fixed nudge mask logic (regen if either *_nudge.gr3 missing)
- Robust NaN-repair ordering (ifix)
- Optional low-pass filter hooks
- **MPI mode**: scatter files to ranks, each rank interpolates its portion, root gathers
  time series, sorts by time, performs time interpolation & writes NetCDFs

Run examples:
  mpirun -n 16 python gen_GM_3Dth_nudge_mpi.py \
      --start 2022-01-02 --end 2022-04-30 --dt 1 \
      --ibnds 1 2 --rlmax_m 30000 --log INFO

  # or edit USER_CFG inside this file and run without flags

Notes:
- Only rank 0 writes output files.
- For very long spans, prefer larger rank counts to reduce per-rank memory.
- This script expects `pylib` utilities (SCHISM helpers) in your PYTHONPATH.
"""

from pylib import *  # zdata, ReadNC, WriteNC, read_schism_hgrid, read_schism_vgrid, loadz, fexist, datenum, lpfilt, near_pts
import os
import argparse
import logging
import numpy as np
from numpy import array, arange, argsort, meshgrid, c_, nonzero
from scipy import interpolate

# --------------------------- MPI setup ---------------------------
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    # Fallback to serial
    class _Dummy:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def bcast(self, x, root=0): return x
        def barrier(self): pass
        def gather(self, x, root=0): return [x]
    COMM = _Dummy()
    RANK = 0
    SIZE = 1

# -----------------------------------------------------------------------------
# USER CONFIG (edit here) + CLI overrides
# -----------------------------------------------------------------------------
USER_CFG = {
    'grd': '../../../grid/01/',
    'dir_data': '/storage/home/hcoda1/4/kpark350/data/dataset/CMEMS/NorthPacific',
    'start': '2022-01-02',
    'end':   '2022-04-30',
    'dt': 1.0,
    'ibnds': [1],
    'ifix': 1,
    'rlmax_m': 20000.0,
    'rnu_day': 0.25,
    # variables (adjust to dataset)
    'svars': ['sla', 'thetao', 'so', ['uo','vo'], 'thetao', 'so'],
    'snames': ['elev2D.th.nc', 'TEM_3D.th.nc', 'SAL_3D.th.nc', 'uv3D.th.nc', 'TEM_nu.nc', 'SAL_nu.nc'],
    'mvars': ['elev',         'temp',        'salt',         ['u','v'],     'temp',     'salt'],
    'iflags': [1, 1, 1, 1, 1, 1],
    'iLP': [0, 0, 0, 0, 0, 0],
    'fc': 0.25,
    'coor': ['longitude', 'latitude', 'depth'],
    'log': 'INFO',
}

# CLI (flags override USER_CFG)
parser = argparse.ArgumentParser(description='Generate SCHISM boundary and nudge from GM/CMEMS grids (MPI-enabled)')
parser.add_argument('--grd', default=None, help='Grid directory with hgrid.ll and vgrid.in or grid.npz')
parser.add_argument('--data', dest='dir_data', default=None, help='Directory of source GM files')
parser.add_argument('--start', default=None, help='Start date (YYYY-MM-DD)')
parser.add_argument('--end',   default=None, help='End date (YYYY-MM-DD)')
parser.add_argument('--dt', type=float, default=None, help='Output time step in days')
parser.add_argument('--ibnds', type=int, nargs='+', default=None, help='Open boundary IDs to include (1-based)')
parser.add_argument('--ifix', type=int, default=None, choices=[0,1], help='0: fix NaNs first then interp; 1: interp then repair parents (default)')
parser.add_argument('--rlmax_m', type=float, default=None, help='Nudge radius (meters)')
parser.add_argument('--rnu_day', type=float, default=None, help='Nudge time scale (days)')
parser.add_argument('--log', default=None, help='Logging level')
args = parser.parse_args()

from types import SimpleNamespace
cfgd = USER_CFG.copy()
for k, v in vars(args).items():
    if v is not None:
        cfgd[k] = v
cfg = SimpleNamespace(**cfgd)

# Root sets logging
if RANK == 0:
    logging.basicConfig(level=getattr(logging, cfg.log.upper(), logging.INFO), format='[%(levelname)s] %(message)s')
else:
    logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

# Broadcast cfg to all ranks (simple dict)
cfg_dict = dict(cfg.__dict__)
cfg_dict = COMM.bcast(cfg_dict, root=0)
cfg = SimpleNamespace(**cfg_dict)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _interp_weights(axis_nodes, targets):
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

# -----------------------------------------------------------------------------
# Resolve time window & variables
# -----------------------------------------------------------------------------
StartT = datenum(int(cfg.start[0:4]), int(cfg.start[5:7]), int(cfg.start[8:10]))
EndT   = datenum(int(cfg.end[0:4]),   int(cfg.end[5:7]),   int(cfg.end[8:10]))
iflags = cfg.iflags
svars  = cfg.svars
snames = cfg.snames
mvars  = cfg.mvars
dts    = [cfg.dt]*len(svars)

iLP = cfg.iLP
fc  = cfg.fc
coor = cfg.coor

# -----------------------------------------------------------------------------
# Discover input files
# -----------------------------------------------------------------------------
dir_data = cfg.dir_data
fnames = []
ftimes = []
if os.path.isfile(f'{dir_data}/dates.out'):
    mti = array(loadtxt(f'{dir_data}/dates.out'))
    fid = open(f'{dir_data}/files.out').read().split()
    fnames = np.array(fid, dtype=object)
else:
    if RANK == 0:
        logging.warning('dates.out/files.out not found; scanning *.nc to infer chronology (slower).')
        import glob
        fcs = sorted(glob.glob(os.path.join(dir_data, '*.nc')))
        for f in fcs:
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
    # broadcast from root
    mti = COMM.bcast(mti if RANK==0 else None, root=0)
    fnames = COMM.bcast(fnames if RANK==0 else None, root=0)

if fnames.size == 0:
    if RANK == 0:
        raise RuntimeError('No input files discovered.')
    else:
        raise SystemExit

# Filter by time window with ±1 day buffer
fpt = (mti >= (StartT - 1)) * (mti < (EndT + 1))
fnames = fnames[fpt]
mti = mti[fpt]
sind = argsort(mti)
mti = mti[sind]
fnames = fnames[sind]

if RANK == 0:
    logging.info(f'Found {len(fnames)} source files spanning {len(mti)} time anchors.')

# -----------------------------------------------------------------------------
# Read SCHISM grid (broadcast to ranks)
# -----------------------------------------------------------------------------
if RANK == 0:
    grddir = cfg.grd
    if fexist(grddir + '/grid.npz'):
        gd = loadz(grddir + '/grid.npz').hgrid
        vd = loadz(grddir + '/grid.npz').vgrid
        gd.x, gd.y = gd.lon, gd.lat
    else:
        gd = read_schism_hgrid(grddir + '/hgrid.ll')
        vd = read_schism_vgrid(grddir + '/vgrid.in')

    # Prepare nudging masks (if needed)
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
    # Vertical target depths
    zcor = compute_zcor(gd, vd)
    lzi = zcor[bind]

    # Broadcast light-weight data needed by all ranks
    root_payload = {
        'gd_x': gd.x, 'gd_y': gd.y, 'bind': bind, 'nobn': nobn,
        'lxi': lxi, 'lyi': lyi, 'lzi': lzi, 'nvrt': nvrt,
    }
else:
    root_payload = None

root_payload = COMM.bcast(root_payload, root=0)

# Unpack
nobn = root_payload['nobn']
lxi  = root_payload['lxi']
lyi  = root_payload['lyi']
lzi  = root_payload['lzi']
nvrt = root_payload['nvrt']
bind = root_payload['bind']

# Clamp lzi to max source depth when reading per-file (we do it inside loop once sz is known)

# -----------------------------------------------------------------------------
# Work per stream
# -----------------------------------------------------------------------------
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

    # Prepare local containers
    local_time = []
    local_series = {mv: [] for mv in mvar}

    # Rank-local caches for interpolation weights
    sx0 = sy0 = sz0 = None
    idx0 = idy0 = ratx0 = raty0 = None
    idx = idy = idz = None
    ratx = raty = ratz = None
    sxy = bxy = None

    # Distribute files across ranks (round-robin)
    my_files = fnames[RANK::SIZE]

    for fname in my_files:
        fp = f'{dir_data}/{fname}'
        try:
            C = ReadNC(fp, 1)
        except Exception as e:
            if RANK == 0:
                logging.error(f'Rank {RANK}: failed to open {fname}: {e}')
            continue

        ctime, reftime = _parse_time_units(C.variables['time'])
        local_time.extend(ctime)

        # Coordinates
        sx = array(C.variables[coor[0]][:])
        sy = array(C.variables[coor[1]][:])
        sz = array(C.variables[coor[2]][:])
        nz = len(sz)

        # Ensure lzi within source depth range (avoid upper-bound index error)
        lzi_clamped = lzi.copy()
        fpz = lzi_clamped >= sz.max()
        lzi_clamped[fpz] = sz.max() - 1e-6

        # Longitude wrap
        lonidx = None
        if sx.max() > 180:
            sx = (sx + 180) % 360 - 180
            lonidx = argsort(sx)
            sx = sx[lonidx]

        # Latitude ensure increasing
        sy, flipped_lat = _ensure_increasing(sy, 'latitude')

        # If grid changed, rebuild weights and NaN-repair index maps
        if (sx0 is None) or (not np.array_equal(sx, sx0)) or (not np.array_equal(sy, sy0)) or (not np.array_equal(sz, sz0)):
            sxi, syi = meshgrid(sx, sy)
            sxy = c_[sxi.ravel(), syi.ravel()]
            bxy = c_[lxi, lyi]

            # Optional repair maps using a sample 3D var if available
            sindns = sindps = None
            if len(svar) > 1:  # e.g., uo/vo style or temp/salt 3D
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
                    fpn = abs(r) > 1e3
                    sindn = nonzero(fpn)[0]
                    sindr = nonzero(~fpn)[0]
                    if len(sindr) != 0:
                        sindp = sindr[near_pts(sxy[sindn], sxy[sindr])]
                    else:
                        sindp = array([])
                    sindns.append(sindn); sindps.append(sindp)

            # Weights for 2D & 3D
            sx0 = sx.copy(); sy0 = sy.copy(); sz0 = sz.copy()
            idx0, ratx0 = _interp_weights(sx0, lxi)
            idy0, raty0 = _interp_weights(sy0, lyi)
            idx, ratx = _interp_weights(sx0, lxi)
            idy, raty = _interp_weights(sy0, lyi)
            idz, ratz = _interp_weights(sz0, lzi_clamped)

        # Loop time in this file
        for i, cti in enumerate(ctime):
            for svari, mvari in zip(svar, mvar):
                cv = array(C.variables[svari][i])
                # Reindex lon/lat orientation
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

                # Entire-field missing → fallback
                if np.sum(np.abs(cv) < 1e3) == 0:
                    last = local_series[mvari][-1] if len(local_series[mvari]) else (np.zeros((nobn,)) if mvari=='elev' else np.zeros((nobn, nvrt)))
                    local_series[mvari].append(last)
                    continue

                # --- Interpolation ---
                if mvari == 'elev':
                    if cfg.ifix == 0 and sxy is not None:
                        cv_flat = cv.ravel()
                        fpn = np.abs(cv_flat) > 1e3
                        if np.any(fpn):
                            fri = np.nonzero(~fpn)[0]
                            fni = np.nonzero(fpn)[0]
                            if len(fri) > 0:
                                cv_flat[fni] = interpolate.griddata(sxy[fri, :], cv_flat[fri], sxy[fni, :], method='nearest')
                        cv = cv_flat.reshape(cv.shape)

                    v0 = np.array([
                        cv[idy0, idx0],
                        cv[idy0, idx0 + 1],
                        cv[idy0 + 1, idx0],
                        cv[idy0 + 1, idx0 + 1]
                    ])
                    if cfg.ifix == 1:
                        for ii in range(4):
                            parent = v0[ii]
                            fpn = np.abs(parent) > 1e3
                            if np.any(fpn):
                                parent[fpn] = interpolate.griddata(bxy[~fpn, :], parent[~fpn], bxy[fpn, :], method='nearest')
                            v0[ii] = parent
                    v1 = v0[0] * (1 - ratx0) + v0[1] * ratx0
                    v2 = v0[2] * (1 - ratx0) + v0[3] * ratx0
                    vi = v1 * (1 - raty0) + v2 * raty0
                else:
                    if cfg.ifix == 0 and isinstance(sindns, list):
                        for ii in arange(nz):
                            if np.sum(np.abs(cv[ii]) < 1e3) == 0 and ii > 0:
                                cv[ii] = cv[ii - 1]
                            sindn, sindp = sindns[ii], sindps[ii]
                            if len(sindp) != 0:
                                cvi = cv[ii].ravel()
                                cvi[sindn] = cvi[sindp]
                                fpn = np.abs(cvi) > 1e3
                                if np.any(fpn):
                                    fri = np.nonzero(~fpn)[0]
                                    fni = np.nonzero(fpn)[0]
                                    if len(fri) > 0:
                                        cvi[fni] = interpolate.griddata(sxy[fri, :], cvi[fri], sxy[fni, :], method='nearest')
                                cv[ii] = cvi.reshape(cv[ii].shape)

                    v0 = np.array([
                        cv[idz, idy, idx],      cv[idz, idy, idx + 1],
                        cv[idz, idy + 1, idx],  cv[idz, idy + 1, idx + 1],
                        cv[idz + 1, idy, idx],  cv[idz + 1, idy, idx + 1],
                        cv[idz + 1, idy + 1, idx], cv[idz + 1, idy + 1, idx + 1]
                    ])
                    if cfg.ifix == 1:
                        for ii in range(8):
                            parent = v0[ii]
                            fpn = np.abs(parent) > 1e3
                            if np.any(fpn):
                                parent[fpn] = interpolate.griddata(bxy[~fpn, :], parent[~fpn], bxy[fpn, :], method='nearest')
                            v0[ii] = parent
                    v1 = v0[0] * (1 - ratx) + v0[1] * ratx
                    v2 = v0[2] * (1 - ratx) + v0[3] * ratx
                    v3 = v0[4] * (1 - ratx) + v0[5] * ratx
                    v4 = v0[6] * (1 - ratx) + v0[7] * ratx
                    v5 = v1 * (1 - raty) + v2 * raty
                    v6 = v3 * (1 - raty) + v4 * raty
                    vi = v5 * (1 - ratz) + v6 * ratz

                local_series[mvari].append(vi)

        C.close()

    # -------------------- Gather to root --------------------
    COMM.barrier()
    all_times = COMM.gather(np.array(local_time, dtype=float), root=0)
    all_series = {}
    for mv in mvar:
        all_series[mv] = COMM.gather(np.array(local_series[mv], dtype=object), root=0)

    if RANK != 0:
        continue

    # -------------------- Root concatenation --------------------
    S_time = np.concatenate(all_times) if len(all_times) else np.array([])
    if S_time.size == 0:
        logging.warning(f'No data gathered for {sname}; skipping')
        continue
    # Stack per variable
    sdict = {'time': S_time}
    for mv in mvar:
        parts = [arr for chunk in all_series[mv] for arr in chunk]
        sdict[mv] = np.stack(parts, axis=0)

    # Sort by time
    sind = np.argsort(sdict['time'])
    sdict['time'] = sdict['time'][sind]
    for mv in mvar:
        sdict[mv] = sdict[mv][sind]

    # Time interpolation → mtime
    mtime = arange(StartT, EndT + dt + 1e-9, dt)
    nt = len(mtime)
    if not (np.min(sdict['time']) <= mtime[0] and np.max(sdict['time']) >= mtime[-1]):
        raise RuntimeError('Source time range does not cover requested mtime range')

    for mv in mvar:
        f = interpolate.interp1d(sdict['time'], sdict[mv], axis=0, bounds_error=False, fill_value=(sdict[mv][0], sdict[mv][-1]))
        svi = f(mtime)
        if iLP[n] == 1:
            svi = lpfilt(svi, dt, fc)
        sdict[mv] = svi

    # Reshape 3D variables
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
        elif sname in ['TEM_3D.th.nc', 'SAL_3D.th.nc']:
            dims = [nobn, nvrt, 1, 1, nt]
            vi = sdict[mvar[0]][..., None]
        nd = zdata(); nd.dimname = dimname; nd.dims = dims
        z = zdata(); z.attrs = ['long_name']; z.long_name = 'time step (sec)'; z.dimname = ('one',); z.val = array(dt * 86400.0); nd.time_step = z
        z = zdata(); z.attrs = ['long_name']; z.long_name = 'time (sec)'; z.dimname = ('time',); z.val = (mtime - mtime[0]) * 86400.0; nd.time = z
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

