#!/usr/bin/env python3
from pylib import *

# --------------------------------------------------------
# inputs
# --------------------------------------------------------
grd = './hgrid.gr3'  # model grid (*.npz, or *.gr3)

# --------------------------------------------------------
# read grid
# --------------------------------------------------------
gd = loadz(grd).hgrid if grd.endswith('.npz') else read_schism_hgrid(grd)
gd.compute_all()

## check fluxflag.prop
clf()
evi=gd.read_prop('./fluxflag.prop')
# gd.plot()
gd.plot(fmt=1,value=evi,cmap='jet',cb=True)
# colorbar(ticks=arange(0,14))
show()
