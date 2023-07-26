from pylib import *
from collections import OrderedDict
close("all")

#------------------------------------------------------------------------------
#input
#------------------------------------------------------------------------------
#grds=['./grid49.npz','./grid44.npz','./grid22.npz','./gridecgom.npz']
#grds=['./grid49.npz','./grid4911sm.npz']
grds=['./grid.npz']
tran='transect.bp.alonggs'
ms=10

bp=read_schism_bpfile(tran)
figure(1,figsize=[18,9])
clf()
for nn,grd in enumerate(grds):
    gd=loadz(grd).hgrid; vd=loadz(grd).vgrid; gd.x,gd.y=gd.lon,gd.lat; nvrt=vd.nvrt
    z=compute_zcor(vd.sigma,gd.dp);
    sindp=near_pts(c_[bp.x,bp.y],c_[gd.x,gd.y]); sindp=array(list(OrderedDict.fromkeys(sindp)));#sindp=unique(sindp)
    zi=z[sindp,:]; 
    for i in arange(len(sindp)): fpn=isnan(zi[i]); zi[i][fpn]=min(zi[i])
    dist=[0,]
    dx=gd.x[sindp];dy=gd.y[sindp]
    for i in arange(len(sindp)-1):
        disti=abs((dx[i+1]-dx[i])+1j*(dy[i+1]-dy[i]))+dist[i]
        dist.append(disti)
    dist=array(dist)

    subplot(1,len(grds),nn+1)
    for k in arange(nvrt): plot(dist,zi[:,k],'k')
    for i in arange(len(sindp)): plot(ones(nvrt)*dist[i],zi[i],'k')
    for k in arange(nvrt): plot(dist,zi[:,0],'r.',ms=ms)
    #setp(gca(),ylim=[zi.min()-1,0.5],xlim=[0,dist.max()])
    title(grds[nn]);
    xlabel('Along transect distance (deg)'); ylabel('Depth (m)')
gcf().tight_layout()
show()
