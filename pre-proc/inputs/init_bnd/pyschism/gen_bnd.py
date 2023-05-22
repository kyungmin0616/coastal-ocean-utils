from datetime import datetime
#import os
#os.environ['USE_PYGEOS'] = '0'
import logging

from pyschism.mesh.hgrid import Hgrid
from pyschism.forcing.hycom.hycom2schism import OpenBoundaryInventory

logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger('pyschism')
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    hgrid=Hgrid.open('../../../grid/05/hgrid.gr3', crs='epsg:4326')
    print(hgrid.bbox)
    vgrid='../../../grid/05/vgrid.in'
    outdir='./'
    start_date=datetime(2016, 1, 1)
    rnday=366

    #Additional elev slope
    #lats=[0, 27, 28, 32, 33, 90] 
    #msl_shifts=[0.65, 0.65, 0.65, 0.65, 0.65, 0.65]

    bnd=OpenBoundaryInventory(hgrid, vgrid)
    #boundary; adjust2D=False to turn off elev slope
    #ocean_bnd_ids: indices start 0 so '0' means first bnd. (another e.g.: [0,1] 
    bnd.fetch_data(outdir, start_date, rnday, elev2D=True, TS=True, UV=True, ocean_bnd_ids=[0,1])
    #bnd.fetch_data(outdir, start_date, rnday, elev2D=True, TS=True, UV=True, adjust2D=True, lats=lats,msl_shifts=msl_shifts,ocean_bnd_ids=[0])
