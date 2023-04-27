from datetime import datetime
from time import time
import pathlib
import logging

from pyschism.forcing.nws.nws2.era5 import ERA5

from pyschism.mesh.hgrid import Hgrid
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logging.captureWarnings(True)

log_level = logging.DEBUG
logging.getLogger('pyschism').setLevel(log_level)

if __name__ == "__main__":
    startdate=datetime(2020, 1, 1)
    rnday=60

    t0=time()
    hgrid=Hgrid.open('../../grid/02/hgrid.ll',crs='EPSG:4326')
    bbox = hgrid.get_bbox('EPSG:4326', output_type='bbox')

    er=ERA5()
    outdir = pathlib.Path('./')
    with open("./sflux_inputs.txt", "w") as f:
        f.write("&sflux_inputs\n/\n")
    er.write(outdir=outdir, start_date=startdate, rnday=rnday, air=True, rad=True, prc=True, bbox=bbox, overwrite=True)
    print(f'It took {(time()-t0)/60} minutes to generate {rnday} days')

