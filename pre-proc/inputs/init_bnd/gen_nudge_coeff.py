#! /usr/bin/env python
from datetime import datetime, timedelta
import logging

from pyschism.mesh import Hgrid, Vgrid
from pyschism.forcing.hycom.hycom2schism import Nudge


logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logging.captureWarnings(True)

log_level = logging.DEBUG
logging.getLogger('pyschism').setLevel(log_level)


if __name__ == '__main__':

    hgrid=Hgrid.open('../../../grid/05/ETOPO/hgrid.gr3', crs='epsg:4326')
    outdir='./'

    nudge=Nudge()
    nudge.gen_nudge(outdir, hgrid)
