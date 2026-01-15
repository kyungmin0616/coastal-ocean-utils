from pyschism.mesh import Hgrid
from pyschism.mesh.fgrid import DragCoefficient

hgrid = Hgrid.open('../../../grid/01/hgrid.gr3', crs='epsg:4326')
depth1=-1.0
depth2=-3.0
bfric_river=0.0025
bfric_land=0.025
fgrid=DragCoefficient.linear_with_depth(hgrid, depth1, depth2, bfric_river, bfric_land)
fgrid.write('./drag.gr3', overwrite=True)
