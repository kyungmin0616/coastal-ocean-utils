from datetime import datetime, timedelta
import os
from time import time
import pathlib
import logging

#from pyschism.forcing import source_sink
from pyschism.forcing.source_sink.nwm import NationalWaterModel, NWMElementPairings
from pyschism.mesh import Hgrid


logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logging.captureWarnings(True)

log_level = logging.DEBUG
logging.getLogger('pyschism').setLevel(log_level)

startdate = datetime(2016, 9, 8)
print(startdate)
rnday = 20
hgrid = Hgrid.open("./hgrid.gr3", crs="epsg:4326")

t0 = time()

sources_pairings = pathlib.Path('./sources.json')
sinks_pairings = pathlib.Path('./sinks.json')
output_directory = pathlib.Path('./')
cache = pathlib.Path(f'./{startdate.strftime("%Y%m%d")}')
cache.mkdir(exist_ok=True, parents=True)

if all([sources_pairings.is_file(), sinks_pairings.is_file()]) is False:
    pairings = NWMElementPairings(hgrid)
    sources_pairings.parent.mkdir(exist_ok=True, parents=True)
    pairings.save_json(sources=sources_pairings, sinks=sinks_pairings)
else:
    pairings = NWMElementPairings.load_json(
        hgrid, 
        sources_pairings, 
        sinks_pairings)
nwm=NationalWaterModel(pairings=pairings, cache=cache)
#nwm=NationalWaterModel(pairings=pairings)

nwm.write(output_directory, hgrid, startdate, rnday, overwrite=True)
print(f'It took {time()-t0} seconds to generate source/sink')
