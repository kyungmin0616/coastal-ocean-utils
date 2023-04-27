from pylib import *

sname='./argo_schism/sp/HYCOM.png'
refTime_argo=datenum(1950,1,1)
refdate='2016-09-08'
dir_mdl='/rcfs/projects/mhk_modeling/dataset/HYCOM/global/2016'
dir_argo='/rcfs/projects/mhk_modeling/dataset/ARGO/Matthew/'
StartT=datenum(2016,9,8); EndT=datenum(2016,10,25)

#variables=['longitude','latitude','depth','thetao','so']
variables=['lon','lat','depth','water_temp','salinity']
xms=[34.5,37.5];yms=[34.5,37.5]; exs = linspace(xms[0],xms[1],10)
xmt=[3,32];ymt=[3,32]; ext = linspace(xmt[0],xmt[1],10)


fs=7

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
rc('font', size=SMALL_SIZE)  # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fnames_mdl=array([i for i in os.listdir(dir_mdl) if i.endswith('.nc')])
mti_mdl=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames_mdl])
fpt=(mti_mdl>=(StartT-1))*(mti_mdl<(EndT+1)); fnames_mdl=fnames_mdl[fpt]; mti_mdl=mti_mdl[fpt]
sind=argsort(mti_mdl); mti_mdl=mti_mdl[sind]; fnames_mdl=fnames_mdl[sind]

# find and sort argo files
fnames=array([i for i in os.listdir(dir_argo) if i.endswith('.nc')])
#mti=array([(i.replace('.','_').split('_')[0]) for i in fnames])
mti=array([(i.replace('.nc','')) for i in fnames])
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

MAEt=[]; Corrt=[]; MEt=[]; RMSEt=[];MAEs=[]; Corrs=[]; MEs=[]; RMSEs=[];
figure(1,figsize=[7.2,3.5])


for fnn,fname in enumerate(fnames):
    print('Working on {}/{}'.format(fnn+1,len(fnames)))
    S = ReadNC('{}/{}'.format(dir_argo, fname))
    stimes=S.JULD.val.data+refTime_argo
    depth=S.PRES_ADJUSTED.val*1.01998
    temp=S.TEMP_ADJUSTED.val
    salt=S.PSAL_ADJUSTED.val
    alon=S.LONGITUDE.val
    alat=S.LATITUDE.val

    for nn,stime in enumerate(stimes):
        if sum(depth[nn,:].mask)==len(depth[nn,:]): continue
        if sum(~depth[nn].mask)<4:continue
        #if len(depth.data[nn,:][~depth.mask[nn,:]])<4: continue
        tidx=abs(mti_mdl-stime).argmin()
        C=ReadNC('{}/{}'.format(dir_mdl,fnames_mdl[tidx]),1)

        if any(C.variables[variables[0]][:]>180):
            sx=array((C.variables[variables[0]][:]+180)%360-180); lonidx=argsort(sxp); sxp=sxp[lonidx]
        else:
            sx=array(C.variables[variables[0]][:])
            lonidx=None
        sy=array(C.variables[variables[1]][:]);
        sz=array(C.variables[variables[2]][:]);
        sxi,syi=meshgrid(sx,sy); 
        dist = (syi - alat[nn])**2+(sxi - alon[nn])**2;
        llidx=unravel_index(dist.argmin(), dist.shape)
        if lonidx is not None:
            temp_mdl=array(C.variables[variables[3]][:,:,:,lonidx][:,:,llidx[0],:][:,:,llidx[1]]); fpt=temp_mdl<=-3e4; temp_mdl[fpt]=NaN; temp_mdl=squeeze(temp_mdl)
            salt_mdl=array(C.variables[variables[4]][:,:,:,lonidx][:,:,llidx[0],:][:,:,llidx[1]]); fpt=salt_mdl<=-3e4; salt_mdl[fpt]=NaN; salt_mdl=squeeze(salt_mdl)
        else:
            temp_mdl=array(C.variables[variables[3]][:,:,llidx[0],:][:,:,llidx[1]]); fpt=temp_mdl<=-3e4; temp_mdl[fpt]=NaN; temp_mdl=squeeze(temp_mdl)
            salt_mdl=array(C.variables[variables[4]][:,:,llidx[0],:][:,:,llidx[1]]); fpt=salt_mdl<=-3e4; salt_mdl[fpt]=NaN; salt_mdl=squeeze(salt_mdl)
        odi=depth.data[nn,:]; fpt=depth.mask[nn,:]==1; odi[fpt]=NaN
        if nanmin(odi)<nanmin(sz) or nanmax(odi)>nanmax(sz):
           ofpt=(odi>=sz.min())*(odi<=sz.max())
        else:
           ofpt=full(len(odi),True)
        odi=odi[ofpt]; osi=salt[nn,ofpt]; otpi=temp[nn,ofpt]; fpt=~isnan(odi)*~isnan(osi)*~isnan(otpi); odi=odi[fpt]; osi=osi[fpt]; otpi=otpi[fpt]
        mtpii = interpolate.interp1d(sz,temp_mdl)(odi)
        msii = interpolate.interp1d(sz,salt_mdl)(odi)
        #otpii = interpolate.interp1d(odi,otpi)(sz)
        #osii = interpolate.interp1d(odi,osi)(sz)
        st_temp1=get_stat(mtpii,otpi);
        st_salt1=get_stat(msii,osi); MEti=mean(mtpii)-mean(otpi); MEsi=mean(msii)-mean(osi)
#        st_temp1=get_stat(temp_mdl,otpii); st_salt1=get_stat(salt_mdl,osii); MEti=mean(temp_mdl)-mean(otpii); MEsi=mean(salt_mdl)-mean(osii)

        RMSEt.append(st_temp1.RMSD); MAEt.append(st_temp1.MAE); Corrt.append(st_temp1.R); MEt.append(MEti)
        RMSEs.append(st_salt1.RMSD); MAEs.append(st_salt1.MAE); Corrs.append(st_salt1.R); MEs.append(MEsi)

        subplot(1,2,1)
        scatter(mtpii,otpi,s=2,c=odi,vmin=0,vmax=1000, marker = 'o', cmap = 'jet_r' )
        xlabel('Model ($^\circ$C)')
        ylabel('Observation ($^\circ$C)')

        subplot(1,2,2)
        scatter(msii,osi,s=2,c=odi,vmin=0,vmax=1000, marker = 'o', cmap = 'jet_r' )
        xlabel('Model (PSU)')
        ylabel('Observation (PSU)')

RMSEs=array(RMSEs); MAEs=array(MAEs); Corrs=array(Corrs); MEs=array(MEs); fpns=~isnan(MAEs)
RMSEt=array(RMSEt); MAEt=array(MAEt); Corrt=array(Corrt); MEt=array(MEt); fpnt=~isnan(MAEt)

subplot(1,2,1)
plot(ext,ext,'k',lw=3) #perfect fit
setp(gca(),xlim=xmt); setp(gca(),ylim=ymt)
text(xmt[0]+0.02*diff(xmt),ymt[0]+0.95*diff(ymt),'R={:0.3f}'.format(Corrt[fpnt].mean(axis=0)),fontsize=fs)
text(xmt[0]+0.02*diff(xmt),ymt[0]+0.85*diff(ymt),'ME={:0.3f}'.format(MEt[fpnt].mean(axis=0)),fontsize=fs)
text(xmt[0]+0.02*diff(xmt),ymt[0]+0.9*diff(ymt),'MAE={:0.3f}'.format(MAEt[fpnt].mean(axis=0)),fontsize=fs)
text(xmt[0]+0.02*diff(xmt),ymt[0]+0.8*diff(ymt),'RMSE={:0.3f}'.format(RMSEt[fpnt].mean(axis=0)),fontsize=fs)
gca().xaxis.grid('on');
gca().yaxis.grid('on')

subplot(1,2,2)
plot(exs,exs,'k',lw=3) #perfect fit
setp(gca(),xlim=xms); setp(gca(),ylim=yms)
text(xms[0]+0.02*diff(xms),yms[0]+0.95*diff(yms),'R={:0.3f}'.format(Corrs[fpns].mean(axis=0)),fontsize=fs)
text(xms[0]+0.02*diff(xms),yms[0]+0.85*diff(yms),'ME={:0.3f}'.format(MEs[fpns].mean(axis=0)),fontsize=fs)
text(xms[0]+0.02*diff(xms),yms[0]+0.9*diff(yms),'MAE={:0.3f}'.format(MAEs[fpns].mean(axis=0)),fontsize=fs)
text(xms[0]+0.02*diff(xms),yms[0]+0.8*diff(yms),'RMSE={:0.3f}'.format(RMSEs[fpns].mean(axis=0)),fontsize=fs)
gca().xaxis.grid('on');
gca().yaxis.grid('on')

gcf().tight_layout()
savefig('{}'.format(sname), bbox_inches='tight')
 

print('done')
