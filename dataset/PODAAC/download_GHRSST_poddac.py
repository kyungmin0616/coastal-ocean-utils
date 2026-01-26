#!/usr/bin/env python3
#Downlaod JPL-GHRSST
from pylib import *


#--------------------------------------------------------------------
#input
#--------------------------------------------------------------------
StartT,EndT=datenum(2022,1,1),datenum(2023,1,1)
#database
url='https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1'; 
pname='090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'


#download hycom data
for n,ti in enumerate(arange(StartT,EndT)):
    
    furl='podaac-data-downloader -c MUR-JPL-L4-GLOB-v4.1 -d ./ --start-date {}T00:00:00Z --end-date {}T00:00:00Z -e .nc'.format(num2date(ti).strftime('%Y-%m-%d'),num2date(ti).strftime('%Y-%m-%d')) 

    #download hycom data
    if os.path.exists('{}{}'.format(num2date(ti).strftime('%Y%m%d'),pname)): continue
    try: 
       os.system(furl)      
       print('{}{}'.format(num2date(ti).strftime('%Y%m%d'),pname))
    except: 
       pass
       print('data exist')
print('Done-----------')
