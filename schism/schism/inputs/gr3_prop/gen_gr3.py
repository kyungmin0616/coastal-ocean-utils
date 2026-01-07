from pylib import *

gd=read_schism_hgrid('../../../grid/05/ETOPO/hgrid.gr3')

#gd.write_hgrid('./estuary.gr3',value=0)
gd.write_hgrid('./drag.gr3',value=0.0025)
gd.write_hgrid('./diffmin.gr3',value=1e-6)
gd.write_hgrid('./diffmax.gr3',value=1)
#gd.write_hgrid('./rough.gr3',value=0.001)
gd.write_hgrid('./albedo.gr3',value=1e-1)
gd.write_hgrid('./watertype.gr3',value=1) 
gd.write_hgrid('./windrot_geo2proj.gr3',value=0)
gd.write_prop('./tvd.prop',value=1)
#gd.write_hgrid('./salt.ic',value=0)
#gd.write_hgrid('./temp.ic',value=20)
#gd.write_hgrid('./elev.ic',value=0)

print('---------done-----')
