#!/usr/bin/env python3
#create hotstart condition based on GM (Global ocean Model) data
from pylib import *
import time
close("all")

#------------------------------------------------------------------------------
#input
#------------------------------------------------------------------------------
StartT=datenum(2022,1,2)
run='./'
dir_data='/Users/kpark/Downloads/'
bad_val=1e3
qc=True
#the threshold used to flag invalid values in the source data. Any value with abs(value) > bad_val (default 1e3) is treated as “bad” (e.g., missing/land values). It’s used during interpolation to detect and repair those points and in QC to compute bad fractions.

#variables to be interpolated
#CMEMS
svars=['thetao','so'] 
coor=['longitude','latitude','depth']
reftime=datenum(1950,1,1)

#HYCOM
#svars=['water_temp','salinity']
#coor=['lon','lat','depth']
#reftime=datenum(2000,1,1)

mvars=['temp','salt']

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
#------------------------------------------------------------------------------
#interpolate GM data to boundary
#------------------------------------------------------------------------------
t0=time.time()

#find GM file
#fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])
#mti=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames])
#fpt=nonzero(abs(mti-StartT)==min(abs(mti-StartT)))[0][0]; fname=fnames[fpt]
fname='mercatorglorys12v1_gl12_mean_202002.nc'
#read grid
if fexist(run+'/grid.npz'):
   gd=loadz(run+'/grid.npz').hgrid; vd=loadz(run+'/grid.npz').vgrid
else:
   gd=read_schism_hgrid(run+'/hgrid.gr3'); vd=read_schism_vgrid(run+'/vgrid.in')

ne,np,ns,nvrt=gd.ne,gd.np,gd.ns,vd.nvrt

#get node xyz
lxi=gd.x; lyi=gd.y; lzi0=abs(vd.compute_zcor(gd.dp)).T

#get GM time, xyz
C=ReadNC('{}/{}'.format(dir_data,fname),1); #print(fname)
ctime=_parse_time_units(C.variables['time'], reftime); sx=array(C.variables[coor[0]][:])
sy=array(C.variables[coor[1]][:]); sz=array(C.variables[coor[2]][:]);
if sz[0] != 0: sz[0]=0
if sx.max()>180:
    print('Convert [0, 360] to [-180, 180]')
    sx=(sx+180)%360-180; lonidx=argsort(sx); sx=sx[lonidx]
else:
    lonidx=None
fpz=lzi0>=sz.max(); lzi0[fpz]=sz.max()-1e-6

#interp for ST
S=zdata()
sdict={mvar: [] for mvar in mvars}

# precompute 2D interp indices for all nodes
idx, ratx = _interp_weights(sx, lxi)
idy, raty = _interp_weights(sy, lyi)

for k in arange(nvrt):
    lzi=lzi0[k]; bxyz=c_[lxi,lyi,lzi]
    # vertical interp indices for this level
    idz, ratz = _interp_weights(sz, lzi)

    for m, svar in enumerate(svars):
        print(svar, k)
        mvar = mvars[m]
        cv = array(C.variables[svar][0])
        if lonidx is not None:  cv=cv[:,:,lonidx]
        v0=array([cv[idz,idy,idx],cv[idz,idy,idx+1],cv[idz,idy+1,idx],cv[idz,idy+1,idx+1],
              cv[idz+1,idy,idx],cv[idz+1,idy,idx+1],cv[idz+1,idy+1,idx],cv[idz+1,idy+1,idx+1]])

        #remove nan pts
        for n in arange(8):
            fpn=abs(v0[n])>bad_val
            v0[n,fpn]=sp.interpolate.griddata(bxyz[~fpn,:],v0[n,~fpn],bxyz[fpn,:],'nearest',rescale=True)

        v11=v0[0]*(1-ratx)+v0[1]*ratx;  v12=v0[2]*(1-ratx)+v0[3]*ratx; v1=v11*(1-raty)+v12*raty
        v21=v0[4]*(1-ratx)+v0[5]*ratx;  v22=v0[6]*(1-ratx)+v0[7]*ratx; v2=v21*(1-raty)+v22*raty
        vi=v1*(1-ratz)+v2*ratz
        sdict[mvar].append(vi)

for mvar in mvars:
    sdict[mvar]=array(sdict[mvar])

if qc:
    for mvar in mvars:
        frac, vmin, vmax = _qc_stats(sdict[mvar], bad_val=bad_val)
        print('QC {}: bad_frac={:.4f} min={:.4g} max={:.4g}'.format(mvar, frac, vmin, vmax))

#------------------------------------------------------------------------------
#creat netcdf
#------------------------------------------------------------------------------
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
