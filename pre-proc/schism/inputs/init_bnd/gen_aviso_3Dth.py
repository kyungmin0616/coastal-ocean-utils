#!/usr/bin/env python2
#create boundary condition based on hycom data
from pylib import *
close("all")

#------------------------------------------------------------------------------
#input
#------------------------------------------------------------------------------
StartT=datenum(2016,9,8,0,0,0); EndT=datenum(2016,11,1,0,0,0); dt=1
grd='../../../grid/05/ETOPO_11SM/grid.npz'
dir_hycom='/rcfs/projects/mhk_modeling/dataset/AVISO/'
iLP=0; fc=0.5  #iLP=1: remove tidal signal with cutoff frequency fc (day)

ifix=0  #ifix=0: fix hycom nan 1st, then interp;  ifix=1: interp 1st, then fixed nan
#------------------------------------------------------------------------------
#interpolate hycom data to boundary
#------------------------------------------------------------------------------
mtime=arange(StartT,EndT+dt,dt); nt=len(mtime)

#variables for each files
snames=['elev2D.th.nc']
svars=['adt']
mvars=['elev']

#find all hycom files
fnames=array([i for i in os.listdir(dir_hycom) if i.endswith('.nc')])
mti=datenum(array([(i.replace('.','_').split('_')[5]) for i in fnames]))
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

#read hgrid
gd=loadz(grd).hgrid; vd=loadz(grd).vgrid; gd.x,gd.y=gd.lon,gd.lat; nvrt=vd.nvrt

#get bnd node xyz
bind=[]; nobn=[]
for i in arange(len(gd.iobn)):
    bind.extend(gd.iobn[i])
bind=array(bind); nobn=len(bind)

lxi0=gd.x[bind]; lyi0=gd.y[bind]; bxy=c_[lxi0,lyi0] #for 2D
if vd.ivcor==2:
    lzi=abs(compute_zcor(vd.sigma,gd.dp[bind],ivcor=2,vd=vd)).ravel()
else:
    lzi=abs(compute_zcor(vd.sigma[bind],gd.dp[bind])).ravel();
sx0,sy0=None,None

#for each variables
for n,sname in enumerate(snames):
    svar=svars[n]; mvar=mvars[n]
    if isinstance(svar,str): svar=[svar]; mvar=[mvar]

    #interp in space
    S=zdata(); S.time=[]
    [exec('S.{}=[]'.format(i)) for i in mvar]
    for m,fname in enumerate(fnames):
        C=ReadNC('{}/{}'.format(dir_hycom,fname),1); print(fname)
        ctime=array(C.variables['time'])+datenum(1950,1,1); sx=array(C.variables['longitude'][:]); sx=(sx+180)%360-180; lonidx=argsort(sx); sx=sx[lonidx] 
        #sx=array(C.variables['longitude'][:])%360
        sy=array(C.variables['latitude'][:]);  
        
        if not array_equal(sx,sx0)*array_equal(sy,sy0):
            #get interp index for HYCOM data
            if ifix==0:
                sxi,syi=meshgrid(sx,sy); sxy=c_[sxi.ravel(),syi.ravel()];
                cvs=array(C.variables['adt'][0]); cvs=cvs[:,lonidx]; sindns=[]; sindps=[]
                sindns=[]; sindps=[]
                print('computing AVISO interpation index')
                cv=cvs; ds=cv.shape; cv=cv.ravel()
                fpn=abs(cv)>1e3; sindn=nonzero(fpn)[0]; sindr=nonzero(~fpn)[0]; sindp=sindr[near_pts(sxy[sindn],sxy[sindr])]
                sindns.append(sindn); sindps.append(sindp)

            #get interp index for pts
            sx0=sx[:]; sy0=sy[:]; print('get new interp indices: {}'.format(fname))
            idx0=((lxi0[:,None]-sx0[None,:])>=0).sum(axis=1)-1; ratx0=(lxi0-sx0[idx0])/(sx0[idx0+1]-sx0[idx0])
            idy0=((lyi0[:,None]-sy0[None,:])>=0).sum(axis=1)-1; raty0=(lyi0-sy0[idy0])/(sy0[idy0+1]-sy0[idy0])

        S.time.extend(ctime)
        for i, cti in enumerate(ctime):
            for k,svari in enumerate(svar):
                exec("cv=array(C.variables['{}'][{}])".format(svari,i)); mvari=mvar[k]
                cv=cv[:,lonidx];
                #interp in space
                if mvari=='elev':
                    #remove HYCOM nan pts
                    if ifix==0:
                        sindn,sindp=sindns[0],sindps[0]
                        cv=cv.ravel(); fpn=(abs(cv[sindn])>1e3)*(abs(cv[sindp])<1e3); cv[sindn]=cv[sindp]; fpn=abs(cv)>1e3 #init fix
                        if sum(fpn)!=0: fni=nonzero(fpn)[0]; fri=nonzero(~fpn)[0]; fpi=fri[near_pts(sxy[fni],sxy[fri])]; cv[fni]=cv[fpi] #final fix
                        #fpn=abs(cv.ravel())>1e3; cv.ravel()[fpn]=sp.interpolate.griddata(sxy[~fpn,:],cv.ravel()[~fpn],sxy[fpn,:],'nearest') #old method
                        cv=cv.reshape(ds)

                    #find parent pts
                    v0=array([cv[idy0,idx0],cv[idy0,idx0+1],cv[idy0+1,idx0],cv[idy0+1,idx0+1]])

                    #remove nan in parent pts
                    if ifix==1:
                        for ii in arange(4):fpn=abs(v0[ii])>1e3; v0[ii,fpn]=sp.interpolate.griddata(bxy[~fpn,:],v0[ii,~fpn],bxy[fpn,:],'nearest')

                    #interp
                    v1=v0[0]*(1-ratx0)+v0[1]*ratx0;  v2=v0[2]*(1-ratx0)+v0[3]*ratx0
                    vi=v1*(1-raty0)+v2*raty0



                #save data
                exec('S.{}.append(vi)'.format(mvari))
        C.close();
    S.time=array(S.time); [exec('S.{}=array(S.{})'.format(i,i)) for i in mvar]

    #interp in time
    for mvari in mvar:
        exec('vi=S.{}'.format(mvari))
        svi=interpolate.interp1d(S.time,vi,axis=0)(mtime)
        if iLP==1: svi=lpfilt(svi,dt,fc) #low-pass
        exec('S.{}=svi'.format(mvari))
    S.time=mtime

    #reshape the data, and save
    [exec('S.{}=S.{}.reshape([{},{},{}])'.format(i,i,nt,nobn,nvrt)) for i in mvar if i!='elev']

    #--------------------------------------------------------------------------
    #create netcdf
    #--------------------------------------------------------------------------
    nd=zdata(); nd.file_format='NETCDF4'

    #define dimensions
    nd.dimname=['nOpenBndNodes', 'nLevels', 'nComponents', 'one', 'time']
    sname=='elev2D.th.nc'
    snvrt=1; ivs=1; vi=S.elev[...,None,None]
    nd.dims=[nobn,snvrt,ivs,1,nt]

    # print(mvar,nd.dims,vi.shape)

    #--time step, time, and time series----
    nd.vars=['time_step', 'time', 'time_series']
    nd.time_step=zdata()
    nd.time_step.attrs=['long_name'];nd.time_step.long_name='time step in seconds'
    nd.time_step.dimname=('one',); nd.time_step.val=array(dt*86400)

    nd.time=zdata()
    nd.time.attrs=['long_name'];nd.time.long_name='simulation time in seconds'
    nd.time.dimname=('time',); nd.time.val=(S.time-S.time[0])*86400

    nd.time_series=zdata()
    nd.time_series.attrs=[]
    nd.time_series.dimname=('time','nOpenBndNodes','nLevels','nComponents')
    nd.time_series.val=vi.astype('float32')

    WriteNC(sname,nd)
print('Done-------------')
