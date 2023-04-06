from pylib import *


#######################################################################################################################
StartT=datenum(2016,9,20); EndT=datenum(2016,10,22)

bp=read_shapefile_data('./08_bnd.shp')
px,py=bp.xy.T
lw=1.5
fs=7

# Argo info
dir_argo='/rcfs/projects/mhk_modeling/dataset/ARGO/Matthew/'
refTime_argo=datenum(1950,1,1)

# SCHISM info
run='../run/RUN13b'
svars=['temp','salt']
refdate_sch=datenum('2016-9-8')
stacks=[15,45]
stacks=arange(stacks[0],stacks[1]+1,1/24)
mti_sch=stacks-1+refdate_sch

# GM info
# CMEMS
sname='./argo_schism/each_profile/SCHISMvsCMEMSvsARGO/'
dir_mdl='/rcfs/projects/mhk_modeling/dataset/CMEMS/NAO/reanalysis/'
variables=['longitude','latitude','depth','thetao','so'] # CMEMS
tags=['CMEMS','SCHISM']

#HYCOM
#sname='./argo_schism/each_profile/SCHISMvsHYCOMvsARGO/'
#dir_mdl='/rcfs/projects/mhk_modeling/dataset/HYCOM/NAO/'
#variables=['lon','lat','depth','water_temp','salinity'] # HYCOM
#tags=['HYCOM','SCHISM']

#######################################################################################################################
# control font size in plot
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
#######################################################################################################################

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

for fnn,fname in enumerate(fnames):
    print('Working on {}/{}'.format(fnn+1,len(fnames)))
    S = ReadNC('{}/{}'.format(dir_argo, fname))
    stimes=S.JULD.val.data+refTime_argo
    depth=S.PRES_ADJUSTED.val*1.01998
    temp=S.TEMP_ADJUSTED.val
    salt=S.PSAL_ADJUSTED.val
    alon=S.LONGITUDE.val
    alat=S.LATITUDE.val
    chargo=inside_polygon(c_[alon,alat],px,py)  
    if sum(chargo)==0: print('Argo is outside of model domain'); continue
    for nn,stime in enumerate(stimes):
        if sum((stime>=StartT)*(stime<=EndT))==0: print('Current Argo is out of target data');continue
        if sum(depth[nn,:].mask)==len(depth[nn,:]): print('Entire prifiles is masked'); continue
        if sum(~depth[nn].mask)<20:print('Few data points of Argo'); continue
        if nanmax(depth[nn])<300:print('Pass profile below 300m');continue
        print('Date of Argo: {}'.format(num2date(stime).strftime('%Y-%m-%d %H:%M:%S')))
        #if len(depth.data[nn,:][~depth.mask[nn,:]])<4: continue

        ## SCHISM part
        tidx_sch=abs(mti_sch-stime).argmin()
        istack=int(stacks[tidx_sch]) 
        C=read_schism_output(run,['zcor',*svars],c_[alon[nn],alat[nn]],istack,fmt=1)
        sch_time=C.time+refdate_sch
        tidx_sch=abs(sch_time-stime).argmin()
        sz_sch=-C.zcor[tidx_sch,:]; temp_sch=C.temp[tidx_sch,:]; salt_sch=C.salt[tidx_sch,:]        

        ## GM part
        tidx=abs(mti_mdl-stime).argmin()
        C=ReadNC('{}/{}'.format(dir_mdl,fnames_mdl[tidx]),1)

        if any(C.variables[variables[0]][:]>180):
            sx=array((C.variables[variables[0]][:]+180)%360-180); lonidx=argsort(sx); sx=sx[lonidx]
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

        # Stats

        # GM part
        odi=depth[nn,:].data; fpt=depth[nn,:].mask==1; odi[fpt]=NaN
        if nanmin(odi)<nanmin(sz) or nanmax(odi)>nanmax(sz):
           ofpt=(odi>=nanmin(sz))*(odi<=nanmax(sz))
        else:
           ofpt=full(len(odi),True)
        odi=odi[ofpt]; osi=salt[nn,ofpt]; otpi=temp[nn,ofpt]; fpt=~isnan(odi); odi=odi[fpt]; osi=osi[fpt]; otpi=otpi[fpt]
        mtpii = interpolate.interp1d(sz,temp_mdl)(odi)
        msii = interpolate.interp1d(sz,salt_mdl)(odi)
        st_temp1=get_stat(mtpii,otpi);
        st_salt1=get_stat(msii,osi); MEti1=mean(mtpii)-mean(otpi); MEsi1=mean(msii)-mean(osi)

        # SCHISM part
        odi=depth[nn,:].data; fpt=depth[nn,:].mask==1; odi[fpt]=NaN
        if nanmin(odi)<nanmin(sz_sch) or nanmax(odi)>nanmax(sz_sch):
           ofpt=(odi>=nanmin(sz_sch))*(odi<=nanmax(sz_sch))
        else:
           ofpt=full(len(odi),True)
        odi=odi[ofpt]; osi=salt[nn,ofpt]; otpi=temp[nn,ofpt]; fpt=~isnan(odi); odi=odi[fpt]; osi=osi[fpt]; otpi=otpi[fpt]
        mtpii_sch = interpolate.interp1d(sz_sch,temp_sch)(odi)
        msii_sch = interpolate.interp1d(sz_sch,salt_sch)(odi)
        st_temp2=get_stat(mtpii_sch,otpi);
        st_salt2=get_stat(msii_sch,osi); MEti2=mean(mtpii_sch)-mean(otpi); MEsi2=mean(msii_sch)-mean(osi)

        # Obs part
        odi=depth[nn,:].data; fpt=depth[nn,:].mask==1; odi[fpt]=NaN
        osi=salt[nn,:]; otpi=temp[nn,:]; fpt=~isnan(odi); odi=odi[fpt]; osi=osi[fpt]; otpi=otpi[fpt]

        # Plot
        figure(1,figsize=[7.2,3.5])
        clf()
        subplot(1,2,1)
        plot(alon,alat,'r+',markersize=10)
        plot(sxi[llidx],syi[llidx],'k.',markersize=5)
        legend(['Obs',*tags])
        plot(px,py,'k')
        xlabel('Longitude')
        ylabel('Latitude')
        subplot(1,2,2)
        plot(otpi,odi,'r',lw=lw)
        plot(temp_mdl,sz,'k',lw=lw)
        plot(temp_sch,sz_sch,'b',lw=lw)
        title('lon:{:.2f}, lat:{:.2f}, obs:{}, GM:{}, SCH:{}'.format(alon[nn],alat[nn],num2date(stime).strftime('%Y-%m-%d %H:%M:%S'),num2date(mti_mdl[tidx]).strftime('%Y-%m-%d %H:%M:%S'),num2date(sch_time[tidx_sch]).strftime('%Y-%m-%d %H:%M:%S')), fontsize=fs, fontweight='bold')
        xm=[nanmin(otpi)-1,nanmax(otpi)+1]; ym=[nanmin(odi), nanmax(odi)]
        setp(gca(),ylim=ym,xlim=xm)
        gca().invert_yaxis()
        text(xm[0]+0.1*diff(xm),ym[0]+0.1*diff(ym),'mdl={}'.format(tags[0]),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.15*diff(ym),'R={:0.3f}'.format(st_temp1.R),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.2*diff(ym),'MAE={:0.3f}'.format(st_temp1.MAE),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.25*diff(ym),'ME={:0.3f}'.format(st_temp1.ME),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.3*diff(ym),'RMSE={:0.3f}'.format(st_temp1.RMSD),fontsize=fs)

        text(xm[0]+0.1*diff(xm),ym[0]+0.5*diff(ym),'mdl={}'.format(tags[1]),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.55*diff(ym),'R={:0.3f}'.format(st_temp2.R),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.6*diff(ym),'MAE={:0.3f}'.format(st_temp2.MAE),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.65*diff(ym),'ME={:0.3f}'.format(st_temp2.ME),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.7*diff(ym),'RMSE={:0.3f}'.format(st_temp2.RMSD),fontsize=fs)

        legend(['Obs',*tags],loc='lower right')
        xlabel('Temperature ($^\circ$C)')
        ylabel('Depth (m)')
        gca().xaxis.grid('on');
        gca().yaxis.grid('on')
        gcf().tight_layout()
        savefig('{}/{}_temp_{}'.format(sname,mti[fnn], num2date(stime).strftime('%Y%m%d%H%M%S')) + '.png', dpi=400, bbox_inches='tight')

        figure(1,figsize=[7.2,3.5])
        clf()
        subplot(1,2,1)
        plot(alon,alat,'r+',markersize=10)
        plot(sxi[llidx],syi[llidx],'b.',markersize=5)
        legend(['Obs',*tags])
        plot(px,py,'k')
        xlabel('Longitude')
        ylabel('Latitude')
        subplot(1,2,2)
        plot(osi,odi,'r',lw=lw)
        plot(salt_mdl,sz,'k',lw=lw)
        plot(salt_sch,sz_sch,'b',lw=lw)
        title('lon:{:.2f}, lat:{:.2f}, obs:{}, GM:{}, SCH:{}'.format(alon[nn],alat[nn],num2date(stime).strftime('%Y-%m-%d %H:%M:%S'),num2date(mti_mdl[tidx]).strftime('%Y-%m-%d %H:%M:%S'),num2date(sch_time[tidx_sch]).strftime('%Y-%m-%d %H:%M:%S')), fontsize=fs, fontweight='bold')

        xm=[nanmin(osi)-1,nanmax(osi)+1]; ym=[nanmin(odi), nanmax(odi)]
        setp(gca(),ylim=ym,xlim=xm)
        gca().invert_yaxis()
        text(xm[0]+0.1*diff(xm),ym[0]+0.1*diff(ym),'mdl={}'.format(tags[0]),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.15*diff(ym),'R={:0.3f}'.format(st_salt1.R),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.2*diff(ym),'MAE={:0.3f}'.format(st_salt1.MAE),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.25*diff(ym),'ME={:0.3f}'.format(st_salt1.ME),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.3*diff(ym),'RMSE={:0.3f}'.format(st_salt1.RMSD),fontsize=fs)

        text(xm[0]+0.1*diff(xm),ym[0]+0.5*diff(ym),'mdl={}'.format(tags[1]),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.55*diff(ym),'R={:0.3f}'.format(st_salt2.R),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.6*diff(ym),'MAE={:0.3f}'.format(st_salt2.MAE),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.65*diff(ym),'ME={:0.3f}'.format(st_salt2.ME),fontsize=fs)
        text(xm[0]+0.1*diff(xm),ym[0]+0.7*diff(ym),'RMSE={:0.3f}'.format(st_salt2.RMSD),fontsize=fs)

        legend(['Obs',*tags],loc='lower right')
        xlabel('Salinity (PSU)')
        ylabel('Depth (m)')
        gca().xaxis.grid('on');
        gca().yaxis.grid('on')
        gcf().tight_layout()

        savefig('{}/{}_salt_{}'.format(sname,mti[fnn], num2date(stime).strftime('%Y%m%d%H%M%S')) + '.png',dpi=400, bbox_inches='tight')


print('done')
sys.exit()
