#!/usr/bin/env python3
#create boundary condition based on hycom data
from pylib import *
close("all")

#------------------------------------------------------------------------------
#input
#------------------------------------------------------------------------------
StartT=datenum(2016,9,8,0,0,0); EndT=datenum(2016,9,8,0,0,0); dt=0
grd='../pre-proc/grid/06/grid.npz'
dir_aviso='/rcfs/projects/mhk_modeling/dataset/AVISO'
fname='dt_global_allsat_phy_l4_20160908.nc'
ehot='./hotstart.nc'
nhot='./cmems_aviso_hotstart.nc'
#------------------------------------------------------------------------------
#interpolate hycom data to boundary
#------------------------------------------------------------------------------

#variables for each files
snames=['elev2D.th.nc']
svars=['adt']
mvars=['elev']

#read hgrid
gd=loadz(grd).hgrid; vd=loadz(grd).vgrid; gd.x,gd.y=gd.lon,gd.lat; nvrt=vd.nvrt

#get bnd node xyz
bind=gd.iobn[0]; nobn=gd.nobn[0]

lxi0=gd.x; lyi0=gd.y; bxy=c_[lxi0,lyi0] #for 2D
sx0,sy0=None,None

#for each variables
n=0
svar=svars[n]; mvar=mvars[n]
if isinstance(svar,str): svar=[svar]; mvar=[mvar]

#interp in space
C=ReadNC('{}/{}'.format(dir_aviso,fname),1); print(fname)
sx=array(C.variables['longitude'][:]); sx=(sx+180)%360-180; lonidx=argsort(sx); sx=sx[lonidx]
sy=array(C.variables['latitude'][:]);  
        
#get interp index for HYCOM data
sxi,syi=meshgrid(sx,sy); sxy=c_[sxi.ravel(),syi.ravel()];
cvs=array(C.variables['adt'][0]); cvs=cvs[:,lonidx]; sindns=[]; sindps=[]
print('computing AVISO interpation index')
cv=cvs; ds=cv.shape; cv=cv.ravel()
fpn=abs(cv)>1e3; sindn=nonzero(fpn)[0]; sindr=nonzero(~fpn)[0]; sindp=sindr[near_pts(sxy[sindn],sxy[sindr])]
sindns.append(sindn); sindps.append(sindp)

#get interp index for pts
sx0=sx[:]; sy0=sy[:]; print('get new interp indices: {}'.format(fname))
idx0=((lxi0[:,None]-sx0[None,:])>=0).sum(axis=1)-1; ratx0=(lxi0-sx0[idx0])/(sx0[idx0+1]-sx0[idx0])
idy0=((lyi0[:,None]-sy0[None,:])>=0).sum(axis=1)-1; raty0=(lyi0-sy0[idy0])/(sy0[idy0+1]-sy0[idy0])

exec("cv=array(C.variables['adt'][0])");cv=cv[:,lonidx]; mvari=mvar
sindn,sindp=sindns[0],sindps[0]
cv=cv.ravel(); fpn=(abs(cv[sindn])>1e3)*(abs(cv[sindp])<1e3); cv[sindn]=cv[sindp]; fpn=abs(cv)>1e3 #init fix
if sum(fpn)!=0: fni=nonzero(fpn)[0]; fri=nonzero(~fpn)[0]; fpi=fri[near_pts(sxy[fni],sxy[fri])]; cv[fni]=cv[fpi] #final fix
cv=cv.reshape(ds)

#find parent pts
v0=array([cv[idy0,idx0],cv[idy0,idx0+1],cv[idy0+1,idx0],cv[idy0+1,idx0+1]])


#interp
v1=v0[0]*(1-ratx0)+v0[1]*ratx0;  v2=v0[2]*(1-ratx0)+v0[3]*ratx0
vi=v1*(1-raty0)+v2*raty0
S=ReadNC('{}'.format(ehot))
S.eta2.val=vi
WriteNC('{}'.format(nhot),S)
