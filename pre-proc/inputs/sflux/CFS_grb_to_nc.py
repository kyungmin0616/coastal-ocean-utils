#!/usr/bin/env python3
from pylib import *

#------------------------------------------------------------------------------------
#inputs: creat narr sflux database
#------------------------------------------------------------------------------------
fvns=['wnd10m','prmsl','tmp2m','q2m','dlwsfc','dswsfc','prate']
tyear=2012
#Load data, and precess data
dir_data='./grb2/{}'.format(tyear)
new_dir_data='./NetCDF/{}'.format(tyear)


#Conversion 
if not os.path.exists(new_dir_data): os.mkdir(new_dir_data)
for i, fvn in enumerate(fvns):
    fnames=array([i for i in os.listdir(dir_data) if i.startswith(fvn+'.cdas1')])
    mti=array([array(i.replace('.','_').split('_')[2]).astype('int') for i in fnames])
    sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

    for nn,fname in enumerate(fnames):
        print('Working {}'.format(fname))
        if os.path.isfile('{}/{}.nc'.format(new_dir_data,fname)): print('{} exist'.format(fname));continue
        fcmd='cdo -f nc copy {}/{} {}/{}.nc'.format(dir_data,fname,new_dir_data,fname)
        os.system(fcmd)
        if fvn=='wnd10m':
           fcmd='ncrename -O -v 10u,u -v 10v,v {}/{}.nc'.format(new_dir_data,fname)
           os.system(fcmd)
        if fvn=='tmp2m':
           fcmd='ncrename -O -v 2t,stmp {}/{}.nc'.format(new_dir_data,fname)
           os.system(fcmd)
        if fvn=='12m':
           fcmd='ncrename -O -v 2sh,spfh {}/{}.nc'.format(new_dir_data,fname)
           os.system(fcmd)
