from datetime import datetime
from time import time
import pathlib

from pyschism.forcing.nws.nws2.era5 import ERA5

from pyschism.mesh.hgrid import Hgrid
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



if __name__ == '__main__':

    startdate=datetime(2021, 6, 15)
    rnday=120

    t0=time()
    hgrid=Hgrid.open('../../../grid/02/hgrid.gr3',crs='EPSG:4326')
    bbox = hgrid.get_bbox('EPSG:4326', output_type='bbox')

    er=ERA5()
    outdir = pathlib.Path('./')
    interval = 1
    er.write(
        outdir=outdir, 
        start_date=startdate, 
        rnday=rnday, 
        air=True,   #sflux_air_1
        rad=True,   #sflux_rad_1
        prc=True,   #sflux_prc_1
        bbox=bbox, 
        output_interval=interval,  #raw data is hourly
        overwrite=True,
        tmpdir=outdir) #default tmpdir is system's tmp (/tmp), it is possbile /tmp has no enough space for large dataset.

    #write sflux_inputs.txt
    with open("./sflux_inputs.txt", "w") as f:
        f.write("&sflux_inputs\n/\n")

    print(f'It took {(time()-t0)/60} minutes to generate {rnday} days')
