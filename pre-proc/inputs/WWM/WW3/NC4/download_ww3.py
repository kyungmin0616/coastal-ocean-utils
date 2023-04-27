# %%
from datetime import datetime, timedelta
from dateutil.rrule import rrule, MONTHLY
from dateutil.relativedelta import relativedelta
import numpy as np
import os

wdir = './'
schism_start_time = datetime(2019, 8, 20)
#schism_end_time = datetime(2019, 9, 19) + timedelta(days=60)
schism_end_time = datetime(2019, 9, 19)

effective_end_time = schism_end_time + relativedelta(day=1)
download_times = [dt for dt in rrule(MONTHLY, dtstart=schism_start_time, until=effective_end_time)]

# %%
vars = ['dir', 'fp', 'hs', 'spr', 't02',]
var_basename = 'WW3-GLOB-30M'
filenames = []
for download_time in download_times:
    year_str = download_time.strftime('%Y')
    month_str = download_time.strftime('%m')
    url_base = f"ftp://ftp.ifremer.fr/ifremer/ww3/HINDCAST/GLOBAL/{year_str}_ECMWF/"
    for var in vars:
        filename = f"{var_basename}_{year_str}{month_str}_{var}.nc"
        if not os.path.exists(f'{wdir}/{filename}'):
            os.system(f"curl {url_base}/{var}/{filename} --output {wdir}/{filename}")
        filenames.append(filename)

# %%
with open(f'{wdir}/bndfiles.dat', 'w') as f:
    for filename in filenames:
        f.write(f'{filename}\n')


# %%
