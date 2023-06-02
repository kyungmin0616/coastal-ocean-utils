from pylib import *

dir_files='../gfs'
sfno='1'
StartT,EndT=datenum(2021,6,1),datenum(2021,12,1)

fnames=array([i for i in os.listdir(dir_files) if i.endswith('.nc')])
mti=array([datenum(i.replace('.','_').split('_')[1][0:4]+'-'+i.replace('.','_').split('_')[1][4:6]+'-'+i.replace('.','_').split('_')[1][6:8]) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1));fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

for m,fname in enumerate(fnames):
    print('Working {}'.format(fname))
    cdm= 'ln -s {}/{} sflux_air_{}.{:04d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)
    cdm= 'ln -s {}/{} sflux_prc_{}.{:04d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)
    cdm= 'ln -s {}/{} sflux_rad_{}.{:04d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)
print('Done---------')

