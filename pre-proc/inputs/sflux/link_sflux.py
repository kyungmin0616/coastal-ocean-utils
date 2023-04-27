from pylib import *

dir_files='../HRRR'
sfno='2'
StartT,EndT=datenum(2016,9,8),datenum(2016,11,1)

fnames=array([i for i in os.listdir(dir_files) if i.endswith('.nc')])
mti=array([datenum(i.replace('.','_').split('_')[1]) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1));fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

for m,fname in enumerate(fnames):
    print('Working {}'.format(fname))
    cdm= 'ln -s {}/{} sflux_air_{}.00{:02d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)
    cdm= 'ln -s {}/{} sflux_prc_{}.00{:02d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)
    cdm= 'ln -s {}/{} sflux_rad_{}.00{:02d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)
print('Done---------')

