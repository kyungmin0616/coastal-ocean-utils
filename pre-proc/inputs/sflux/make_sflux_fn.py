#!/usr/bin/env python3
from pylib import *

#------------------------------------------------------------------------------------
#inputs: creat narr sflux database
#------------------------------------------------------------------------------------
tlist=arange(201301,201401,1)
sdir='./'
xl=[-100, 20];yl=[-30, 60]; # lon and lat for subset

#ftp info
dir_data='../NetCDF/2013/'
vars=['air','rad','prc']
svars=[('uwind','vwind','prmsl','stmp','spfh'),('dlwrf','dswrf'),('prate',)] #sflux variable
nvars=[('u','v','prmsl','stmp','spfh'),('dlwrf','dswrf'),('prate',)] #narr variables
fvns=[('wnd10m','wnd10m','prmsl','tmp2m','q2m'),('dlwsfc','dswsfc'),('prate',)]


#Load data, and precess data

cti=0
for fti in tlist:
    cti=cti+1
    #create folder
    #sdir='{}'.format(year);
    #if not os.path.exists(sdir): os.mkdir(sdir)

    #pre-calculation
    S0=loadz('sflux_template.npz');
    # days=arange(datenum(year,1,1),datenum(year+1,1,1))

    #for each dataset
    for m in arange(len(vars)):
        vari=vars[m]; svari=svars[m]; nvari=nvars[m]; fvni=fvns[m]

        #read the data
        C=zdata()
        for svarii,nvarii,fvnii in zip(svari,nvari,fvni):
            print(svarii, nvarii, fvnii)
            exec('C.{}=ReadNC("{}/{}.cdas1.{}.grb2.nc")'.format(svarii,dir_data,fvnii,fti))
            ### correct data
            # invert latitude and data
            exec('C.{}.lat.val=flip(C.{}.lat.val)'.format(svarii,svarii))
            exec('C.{}.{}.val=flip(squeeze(C.{}.{}.val),1)'.format(svarii,nvarii.split('.')[0],svarii,nvarii.split('.')[0])) 
            # Longitude transform
            exec('C.{}.lon.val=(C.{}.lon.val+180)%360-180; sindp=argsort(C.{}.lon.val); C.{}.lon.val=C.{}.lon.val[sindp]'.format(svarii,svarii,svarii,svarii,svarii)) 
            exec('C.{}.{}.val=squeeze(C.{}.{}.val)[:,:,sindp]'.format(svarii,nvarii.split('.')[0],svarii,nvarii.split('.')[0])) 
            # Subset
            exec('loidx=(xl[0]-1<C.{}.lon.val)*(C.{}.lon.val<xl[1]+1); laidx=(yl[0]-1<C.{}.lat.val)*(C.{}.lat.val<yl[1]+1)'.format(svarii,svarii,svarii,svarii))
            exec('C.{}.lon.val=C.{}.lon.val[loidx]; C.{}.lat.val=C.{}.lat.val[laidx]'.format(svarii,svarii,svarii,svarii))
            exec('C.{}.{}.val=C.{}.{}.val[:,:,loidx][:,laidx,:]'.format(svarii,nvarii.split('.')[0],svarii,nvarii.split('.')[0]))

        #processing data and write sflux
        exec('S=S0.{}'.format(vari))
        exec('time=datenum(2020,{},1,0,0,0)+array(C.{}.time.val).astype("float64")/24-(0.5/24)'.format(cti,svari[0]))
        exec('lon=array(C.{}.lon.val); lat=array(C.{}.lat.val)'.format(svari[0],svari[0]))

        # Find different resolution and fix it
        for svarii,nvarii in zip(svari,nvari):
            exec('chlon=C.{}.lon.val; chlat=C.{}.lat.val'.format(svarii,svarii))
            if  len(lon)==len(chlon) and len(lat)==len(chlat): continue
            print('{} has different resolution. Base: {} x {}, {}: {} x {}'.format(svarii,len(lon),len(lat),svarii,len(chlon),len(chlat)));
            exec('ct=C.{}.time.val; clon=C.{}.lon.val; clat=C.{}.lat.val'.format(svarii,svarii,svarii))
            exec('chtmp=C.{}.{}.val'.format(svarii,nvarii))
            newtmp=[]
            for nn in arange(shape(ct)[0]):
                F = interpolate.RectBivariateSpline(chlat,chlon,chtmp[nn,:,:])
                newtmp.append(F(lat,lon))
            exec('C.{}.{}.val=array(newtmp)'.format(svarii,nvarii))

        # mesh
        lon,lat=meshgrid(lon,lat) 
        
        #get days
        days=unique(time.astype('int'))
        #nt,nx,ny=[int(len(time)/len(days)),len(xgrid),len(ygrid)]
        nt,nx,ny=[24,shape(lon)[1],shape(lon)[0]]
        for dayi in days:
            ti=num2date(dayi)
            fp=(time>=dayi)*(time<(dayi+1));

            #dims
            S.dims=[nx,ny,nt]
            #time, lon, lat
            S.time.base_date=array([ti.year,ti.month,ti.day,0]);
            S.time.units='days since {}'.format(ti.strftime('%Y-%m-%d'))
            S.time.dims=[nt]; S.time.val=time[fp]+(0.5/24)-dayi;
            S.lon.dims=[ny,nx]; S.lon.val=lon
            S.lat.dims=[ny,nx]; S.lat.val=lat
            #variables
            for svarii,nvarii in zip(svari,nvari):
                exec('S.{}.dims=[nt,ny,nx]'.format(svarii));
                exec('S.{}.val=C.{}.{}.val[fp,:,:]'.format(svarii,svarii,nvarii.split('.')[0]));

            S.file_format='NETCDF4'
            #write narr files
            fname='CFSV2_{}_{}.nc'.format(vari,ti.strftime('%Y_%m_%d'))
            print('writing {}'.format(fname))
            WriteNC('{}/{}'.format(sdir,fname),S)
print('----------Done---------')
#------------------------------------------------------------------------------
##--prepare template for sflux based on former sflux files
#------------------------------------------------------------------------------
#S=zdata();
#svars=['air','rad','prc']
#for svar in svars:
#    fname='sflux_{}_1.0001.nc'.format(svar)
#    Si=ReadNC(fname,2);
#    #clear variables
#    for vari in Si.vars:
#        exec('Si.{}.val=None'.format(vari))
#    exec('S.{}=Si'.format(svar));
#S.vars=svars;
#savez('sflux_template',S);
