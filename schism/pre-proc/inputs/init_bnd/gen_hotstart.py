#!/usr/bin/env python3
"""
Generate SCHISM hotstart.nc from GM/CMEMS/HYCOM data using MPI over vertical levels.

How to use:
- Edit StartT, run, dir_data, and dataset variable names below.
- Run with mpirun/mpiexec; only rank 0 writes hotstart.nc.
"""
from pylib import *
import time

# -----------------------------------------------------------------------------
# input
# -----------------------------------------------------------------------------
StartT=datenum(2022,1,2)
run='/S/data00/G6008/d1041/Projects/SendaiOnagawa/pre-proc/grid/01/'
dir_data='/S/data00/G6008/d1041/dataset/CMEMS/daily/EastAsia/'
bad_val=1e3
qc=True
# bad_val: threshold for invalid values; qc: print basic min/max and bad fraction.

# variables to be interpolated
# CMEMS
svars=['thetao','so']
coor=['longitude','latitude','depth']
reftime=datenum(1950,1,1)

# HYCOM
#svars=['water_temp','salinity']
#coor=['lon','lat','depth']
#reftime=datenum(2000,1,1)

mvars=['temp','salt']

# -----------------------------------------------------------------------------
# local modification (regions-based, optional)
# -----------------------------------------------------------------------------
use_regions_mod = True
regions_gr3 = 'regions.gr3'
region_values = {
    1: {'temp': 8.0,  'salt': 24.0},
    2: {'temp': 10.0, 'salt': 30.0},
    3: {'temp': 9.0,  'salt': 20.0},
    4: {'temp': 7.0,  'salt': 15.0},
}
blend_dist = 50000.0  # meters
blend_mode = 'cosine'  # 'linear' or 'cosine'
apply_top_layers = None  # e.g., 5 for top 5 layers; None for all
dist_mode = 'meters'  # 'meters' (haversine lon/lat) or 'xy' for projected coords

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _interp_weights(axis_nodes, targets):
    import numpy as _np
    axis_nodes = _np.asarray(axis_nodes)
    targets = _np.asarray(targets)
    n = axis_nodes.size
    # right insertion index minus 1 = lower bin index
    idx = _np.searchsorted(axis_nodes, targets, side='right') - 1
    # clip so idx+1 is always valid
    idx = _np.clip(idx, 0, max(n-2, 0))
    den = axis_nodes[idx+1] - axis_nodes[idx]
    w = _np.zeros_like(targets, dtype=float)
    ok = den != 0
    w[ok] = (targets[ok] - axis_nodes[idx[ok]]) / den[ok]
    # exact-edge and out-of-range behavior
    if n > 0:
        w = _np.where(targets <= axis_nodes[0], 0.0, w)
        w = _np.where(targets >= axis_nodes[-1], 1.0, w)
    return idx, w

def _parse_time_units(nc_time_var, fallback_reftime):
    vals = array(nc_time_var[:], dtype=float)
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

def _qc_stats(arr, bad_val=1e3):
    a = array(arr, dtype=float)
    bad = isnan(a) | (abs(a) > bad_val)
    frac = float(bad.sum()) / float(a.size) if a.size else 0.0
    if bad.all():
        return frac, nan, nan
    return frac, float(nanmin(a[~bad])), float(nanmax(a[~bad]))

def _haversine_dist(lon1, lat1, lon2, lat2):
    r = 6371000.0
    to_r = pi / 180.0
    lon1 = lon1 * to_r; lat1 = lat1 * to_r
    lon2 = lon2 * to_r; lat2 = lat2 * to_r
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * r * arcsin(sqrt(a))

def _blend_weights(dist, max_dist, mode='linear'):
    w = clip(dist / max_dist, 0.0, 1.0)
    if mode == 'cosine':
        w = 0.5 * (1 - cos(pi * w))
    return w

def _apply_regions_mod(temp, salt, gd, bind, lxi, lyi, nvrt):
    if not use_regions_mod:
        return temp, salt
    if not os.path.isfile(regions_gr3):
        print(f'WARNING: regions file not found: {regions_gr3}; skip regions mod')
        return temp, salt
    rg = read_schism_hgrid(regions_gr3)
    rid = around(rg.dp).astype(int)
    if rid.size != gd.x.size:
        print('WARNING: regions.gr3 size mismatch; skip regions mod')
        return temp, salt

    # Only apply to nodes with a configured region value.
    valid_regions = [r for r, v in region_values.items() if v.get('temp') is not None or v.get('salt') is not None]
    if len(valid_regions) == 0:
        return temp, salt

    # Precompute nearest other-region distance and neighbor region id.
    xy = lxi + 1j * lyi
    rid_b = rid[bind]
    neighbor_region = full(bind.shape, -1, dtype=int)
    min_dist = full(bind.shape, inf, dtype=float)

    for r in valid_regions:
        idx_r = where(rid_b == r)[0]
        if idx_r.size == 0:
            continue
        pts_r = xy[idx_r]
        for r2 in valid_regions:
            if r2 == r:
                continue
            idx_o = where(rid_b == r2)[0]
            if idx_o.size == 0:
                continue
            pts_o = xy[idx_o]
            nn = near_pts(pts_r, pts_o)
            dist = abs(pts_r - pts_o[nn])
            # Update nearest neighbor region for r nodes
            upd = dist < min_dist[idx_r]
            min_dist[idx_r[upd]] = dist[upd]
            neighbor_region[idx_r[upd]] = r2

    # Convert distance to meters if needed
    if dist_mode == 'meters':
        # Approximate using haversine with nearest neighbor coordinates
        # Recompute distances for nodes with valid neighbor region
        sel = neighbor_region >= 0
        if any(sel):
            nnr = neighbor_region[sel]
            x0 = lxi[sel]; y0 = lyi[sel]
            x1 = zeros_like(x0); y1 = zeros_like(y0)
            for r2 in valid_regions:
                idx_o = where(rid_b == r2)[0]
                if idx_o.size == 0:
                    continue
                pts_o = xy[idx_o]
                s = where(nnr == r2)[0]
                if s.size == 0:
                    continue
                nn = near_pts(xy[sel][s], pts_o)
                x1[s] = lxi[idx_o[nn]]
                y1[s] = lyi[idx_o[nn]]
            min_dist[sel] = _haversine_dist(x0, y0, x1, y1)

    w = _blend_weights(min_dist, blend_dist, blend_mode)

    # Apply blend to temp/salt
    temp_mod = temp.copy()
    salt_mod = salt.copy()
    k_slice = slice(None)
    if apply_top_layers is not None and apply_top_layers > 0:
        k_slice = slice(nvrt - apply_top_layers, nvrt)

    for r in valid_regions:
        idx_r = where(rid_b == r)[0]
        if idx_r.size == 0:
            continue
        rtemp = region_values[r].get('temp')
        rsalt = region_values[r].get('salt')
        # neighbor region target values
        nbr = neighbor_region[idx_r]
        nbr_temp = array([region_values.get(rr, {}).get('temp') for rr in nbr], dtype=float)
        nbr_salt = array([region_values.get(rr, {}).get('salt') for rr in nbr], dtype=float)
        wr = w[idx_r]

        if rtemp is not None and any(nbr_temp == nbr_temp):
            t0 = rtemp
            t1 = where(isnan(nbr_temp), rtemp, nbr_temp)
            temp_mod[k_slice, idx_r] = (wr * t0 + (1 - wr) * t1)[None, :]
        if rsalt is not None and any(nbr_salt == nbr_salt):
            s0 = rsalt
            s1 = where(isnan(nbr_salt), rsalt, nbr_salt)
            salt_mod[k_slice, idx_r] = (wr * s0 + (1 - wr) * s1)[None, :]

    return temp_mod, salt_mod

# -----------------------------------------------------------------------------
# MPI setup
# -----------------------------------------------------------------------------
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    class _Dummy:
        def Get_rank(self): return 0
        def Get_size(self): return 1
        def barrier(self): pass
        def gather(self, x, root=0): return [x]
    COMM = _Dummy()
    RANK = 0
    SIZE = 1

# -----------------------------------------------------------------------------
# interpolate GM data to grid (parallel over vertical levels)
# -----------------------------------------------------------------------------
t0=time.time()

# find GM file
fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])
mti=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames])
fpt=nonzero(abs(mti-StartT)==min(abs(mti-StartT)))[0][0]; fname=fnames[fpt]

# read grid
if fexist(run+'/grid.npz'):
   gd=loadz(run+'/grid.npz').hgrid; vd=loadz(run+'/grid.npz').vgrid
else:
   gd=read_schism_hgrid(run+'/hgrid.gr3'); vd=read_schism_vgrid(run+'/vgrid.in')

ne,np,ns,nvrt=gd.ne,gd.np,gd.ns,vd.nvrt

# get node xyz
lxi=gd.x; lyi=gd.y; lzi0=abs(vd.compute_zcor(gd.dp)).T

# get GM time, xyz
C=ReadNC('{}/{}'.format(dir_data,fname),1)
ctime=_parse_time_units(C.variables['time'], reftime)
sx=array(C.variables[coor[0]][:]); sy=array(C.variables[coor[1]][:]); sz=array(C.variables[coor[2]][:])
if sz[0] != 0: sz[0]=0
if sx.max()>180:
    if RANK == 0: print('Convert [0, 360] to [-180, 180]')
    sx=(sx+180)%360-180; lonidx=argsort(sx); sx=sx[lonidx]
else:
    lonidx=None
fpz=lzi0>=sz.max(); lzi0[fpz]=sz.max()-1e-6

# precompute horizontal weights
idx, ratx = _interp_weights(sx, lxi)
idy, raty = _interp_weights(sy, lyi)

# load variables once
cv_temp = array(C.variables[svars[0]][0])
cv_salt = array(C.variables[svars[1]][0])
if lonidx is not None:
    cv_temp = cv_temp[:, :, lonidx]
    cv_salt = cv_salt[:, :, lonidx]

# split levels among ranks
all_k = array(range(nvrt))
local_k = all_k[RANK::SIZE]
if RANK == 0:
    print('MPI ranks: {} | total levels: {}'.format(SIZE, nvrt))
print('Rank {} handles levels: {}..{} ({} levels)'.format(
    RANK, local_k[0] if len(local_k) else -1, local_k[-1] if len(local_k) else -1, len(local_k)
))
local_temp = zeros([len(local_k), np])
local_salt = zeros([len(local_k), np])

for ii, k in enumerate(local_k):
    if ii == 0 or ii % 5 == 0 or ii == len(local_k) - 1:
        print('Rank {} progress: level {}/{}'.format(RANK, k, nvrt-1))
    lzi=lzi0[k]; bxyz=c_[lxi,lyi,lzi]
    idz, ratz = _interp_weights(sz, lzi)

    for vname, cv, store in (('temp', cv_temp, local_temp), ('salt', cv_salt, local_salt)):
        print('Rank {} var {} level {}'.format(RANK, vname, k))
        v0=array([cv[idz,idy,idx],cv[idz,idy,idx+1],cv[idz,idy+1,idx],cv[idz,idy+1,idx+1],
              cv[idz+1,idy,idx],cv[idz+1,idy,idx+1],cv[idz+1,idy+1,idx],cv[idz+1,idy+1,idx+1]])
        for n in arange(8):
            fpn=abs(v0[n])>bad_val
            v0[n,fpn]=sp.interpolate.griddata(bxyz[~fpn,:],v0[n,~fpn],bxyz[fpn,:],'nearest',rescale=True)
        v11=v0[0]*(1-ratx)+v0[1]*ratx;  v12=v0[2]*(1-ratx)+v0[3]*ratx; v1=v11*(1-raty)+v12*raty
        v21=v0[4]*(1-ratx)+v0[5]*ratx;  v22=v0[6]*(1-ratx)+v0[7]*ratx; v2=v21*(1-raty)+v22*raty
        store[ii]=v1*(1-ratz)+v2*ratz

# gather to root
payload = {'k': local_k, 'temp': local_temp, 'salt': local_salt}
all_payload = COMM.gather(payload, root=0)
COMM.barrier()

if RANK != 0:
    raise SystemExit

# assemble full arrays on root
temp_all = zeros([nvrt, np])
salt_all = zeros([nvrt, np])
for p in all_payload:
    for i, k in enumerate(p['k']):
        temp_all[k] = p['temp'][i]
        salt_all[k] = p['salt'][i]

sdict = {'temp': temp_all, 'salt': salt_all}
if qc:
    for mvar in mvars:
        frac, vmin, vmax = _qc_stats(sdict[mvar], bad_val=bad_val)
        print('QC {}: bad_frac={:.4f} min={:.4g} max={:.4g}'.format(mvar, frac, vmin, vmax))

# Optional local modification after interpolation
bind_all = arange(np)
sdict['temp'], sdict['salt'] = _apply_regions_mod(sdict['temp'], sdict['salt'], gd, bind_all, gd.x, gd.y, nvrt)

# -----------------------------------------------------------------------------
# create netcdf (root only)
# -----------------------------------------------------------------------------
tr_nd=r_[sdict['temp'][None,...],sdict['salt'][None,...]].T; tr_el=tr_nd[gd.elnode[:,:3]].mean(axis=1)

nd=zdata(); nd.file_format='NETCDF4'
nd.dimname=['node','elem','side','nVert','ntracers','one']; nd.dims=[np,ne,ns,nvrt,2,1]

#--time step, time, and time series----
nd.vars=['time','iths','ifile','idry_e','idry_s','idry','eta2','we','tr_el',
  'tr_nd','tr_nd0','su2','sv2','q2','xl','dfv','dfh','dfq1','dfq2','nsteps_from_cold','cumsum_eta']

vi=zdata(); vi.dimname=('one',); vi.val=array(0.0); nd.time=vi
vi=zdata(); vi.dimname=('one',); vi.val=array(0).astype('int'); nd.iths=vi
vi=zdata(); vi.dimname=('one',); vi.val=array(1).astype('int'); nd.ifile=vi
vi=zdata(); vi.dimname=('one',); vi.val=array(0).astype('int'); nd.nsteps_from_cold=vi

vi=zdata(); vi.dimname=('elem',); vi.val=zeros(ne).astype('int32'); nd.idry_e=vi #idry_e
vi=zdata(); vi.dimname=('side',); vi.val=zeros(ns).astype('int32'); nd.idry_s=vi #idry_s
vi=zdata(); vi.dimname=('node',); vi.val=zeros(np).astype('int32'); nd.idry=vi   #idry
vi=zdata(); vi.dimname=('node',); vi.val=zeros(np); nd.eta2=vi                   #eta2
vi=zdata(); vi.dimname=('node',); vi.val=zeros(np); nd.cumsum_eta=vi             #cumsum_eta

vi=zdata(); vi.dimname=('elem','nVert'); vi.val=zeros([ne,nvrt]); nd.we=vi   #we
vi=zdata(); vi.dimname=('side','nVert'); vi.val=zeros([ns,nvrt]); nd.su2=vi  #su2
vi=zdata(); vi.dimname=('side','nVert'); vi.val=zeros([ns,nvrt]); nd.sv2=vi  #sv2
vi=zdata(); vi.dimname=('node','nVert'); vi.val=zeros([np,nvrt]); nd.q2=vi   #q2
vi=zdata(); vi.dimname=('node','nVert'); vi.val=zeros([np,nvrt]); nd.xl=vi   #xl
vi=zdata(); vi.dimname=('node','nVert'); vi.val=zeros([np,nvrt]); nd.dfv=vi  #dfv
vi=zdata(); vi.dimname=('node','nVert'); vi.val=zeros([np,nvrt]); nd.dfh=vi  #dfh
vi=zdata(); vi.dimname=('node','nVert'); vi.val=zeros([np,nvrt]); nd.dfq1=vi #dfq1
vi=zdata(); vi.dimname=('node','nVert'); vi.val=zeros([np,nvrt]); nd.dfq2=vi #dfq2

vi=zdata(); vi.dimname=('elem','nVert','ntracers'); vi.val=tr_el; nd.tr_el=vi  #tr_el
vi=zdata(); vi.dimname=('node','nVert','ntracers'); vi.val=tr_nd; nd.tr_nd=vi  #tr_nd
vi=zdata(); vi.dimname=('node','nVert','ntracers'); vi.val=tr_nd; nd.tr_nd0=vi #tr_nd0

WriteNC('hotstart.nc',nd)
print('Total runtime: {:.2f} s'.format(time.time()-t0))
