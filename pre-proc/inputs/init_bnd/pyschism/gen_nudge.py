#! /usr/bin/env python
from datetime import datetime, timedelta
import logging

from pyschism.mesh import Hgrid, Vgrid
from pyschism.forcing.hycom.hycom2schism import Nudge


logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger('pyschism')
logger.setLevel(logging.INFO)


if __name__ == '__main__':

    hgrid=Hgrid.open('../../../grid/05/hgrid.gr3', crs='epsg:4326')
    print(hgrid.bbox)
    vgrid='../../../grid/05/vgrid.in'
    outdir='./'
    start_date=datetime(2016, 1, 1)
    rnday=366

    nudge=Nudge(hgrid=hgrid, ocean_bnd_ids=[0,1])
    #rlmax,rnu_day same as FORT script gen_nudge2.f90: rlmax in m or deg; rnu_day is max relax strength in days
    nudge.fetch_data(outdir, vgrid, start_date, rnday, rlmax=0.20, rnu_day=0.25)
    #nudge.fetch_data(outdir, vgrid, start_date, rnday, restart=False, rlmax=1, rnu_day=1)
