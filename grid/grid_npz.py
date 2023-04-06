from pylib import *

gd=read_schism_hgrid('./hgrid.gr3')
gd.lon,gd.lat=gd.x,gd.y #better method: gd.lon,gd.lat=proj_pts(...)
vd=read_schism_vgrid('./vgrid.in')
s=zdata(); s.hgrid=gd; s.vgrid=vd; savez('./grid.npz',s)


