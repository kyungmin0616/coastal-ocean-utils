from pylib import *

dir_files='../../../pre-proc/sflux/sflux-CFSV2-NAO/'
sfno='1'
StartT,EndT=datenum(2020,1,1),datenum(2020,2,29)


fnames=array([i for i in os.listdir(dir_files) if i.startswith('CFSV2_air')])
mti=array([datenum(*array(i.replace('.','_').split('_')[2:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1));fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

for m,fname in enumerate(fnames):
    cdm= 'ln -s {}/{} sflux_air_{}.00{:02d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)

fnames=array([i for i in os.listdir(dir_files) if i.startswith('CFSV2_rad')])
mti=array([datenum(*array(i.replace('.','_').split('_')[2:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1));fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

for m,fname in enumerate(fnames):
    cdm= 'ln -s {}/{} sflux_rad_{}.00{:02d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)

fnames=array([i for i in os.listdir(dir_files) if i.startswith('CFSV2_prc')])
mti=array([datenum(*array(i.replace('.','_').split('_')[2:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1));fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

for m,fname in enumerate(fnames):
    cdm= 'ln -s {}/{} sflux_prc_{}.00{:02d}.nc'.format(dir_files,fname,sfno,m+1)
    os.system(cdm)



print('Done---------')

