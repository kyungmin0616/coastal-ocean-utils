#!/usr/bin/env python3
from pylib import *

sms2grd('./ECGOM_GS_VER2.2dm','./hgrid.ll') # convert from SMS to SCHISM grid
#proj('./hgrid.utm',0,'epsg:26918','./hgrid.ll',0,'epsg:4326')
#proj('./hgrid.gr3',0,'epsg:4326','./hgrid.utm',0,'epsg:26918')
#proj('./hgrid.gr3',0,'epsg:4326','./hgrid2.utm',0,'epsg:32632') # convert from WSG to UTM coordinatproj('./hgrid.gr3',0,'epsg:4326','./hgrid2.utm',0,'epsg:269R2') # convert from WSG to UTM coordinate
#proj('./hgrid.gr3',0,'epsg:4326','./hgrid.utm',0,'epsg:26920') # NAD98 UTM 20; convert from WSG to UTM coordinate
#gd=read_schism_hgrid('./hgrid.utm')

#figure(1)
#clf()
#gd.plot_bnd()
#show()

#gd.compute_all()

#figure(figsize=[18,12])
#method 1: grid
#subplot(1,2,1)
#gd.plot()

#method 2: bathymetry
#subplot(1,2,2)
#gd.plot(fmt=1)

#gcf().tight_layout()
#show()
