from pylib import *
matplotlib.use('Agg')

sname='./image_argo/'
run='../outputs/RUN04a-HRRR/'
run2='../outputs/RUN05b-wo-tide/'
tags=['STOFS3D','KMP']
refTime_argo=datenum(1950,1,1)
stacks=[23,43]
istacks=[*arange(stacks[0],stacks[1]+1)]
refdate='2016-09-08'

SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
rc('font', family='Helvetica')
rc('font', size=SMALL_SIZE)  # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

bp=read_shapefile_data('./hgrid_bnd.shp')
px,py=bp.xy.T

# find and sort argo files
dir_argo='/storage/home/hcoda1/4/kpark350/data/dataset/ARGO/Matthew/'
fnames=array([i for i in os.listdir(dir_argo) if i.endswith('.nc')])
#mti=array([(i.replace('.','_').split('_')[0]) for i in fnames])
mti=array([(i.replace('.nc','')) for i in fnames])
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

for istack in istacks:
    print('reading stack= {}'.format(istack))
    C=ReadNC('{}/out2d_{}.nc'.format(run,istack))
    ctime = array(C.time.val)/3600/24+datenum(refdate);
    startT=ctime.min()
    endT=ctime.max()
    C2=ReadNC('{}/zCoordinates_{}.nc'.format(run,istack))
    C3=ReadNC('{}/temperature_{}.nc'.format(run,istack))
    C4=ReadNC('{}/salinity_{}.nc'.format(run,istack))
    D=ReadNC('{}/out2d_{}.nc'.format(run2,istack))
    D2=ReadNC('{}/zCoordinates_{}.nc'.format(run2,istack))
    D3=ReadNC('{}/temperature_{}.nc'.format(run2,istack))
    D4=ReadNC('{}/salinity_{}.nc'.format(run2,istack))

    for fnn,fname in enumerate(fnames):
        S = ReadNC('{}/{}'.format(dir_argo, fname))
        stime=S.JULD.val+refTime_argo
        sindp=(stime>=startT)*(stime<=endT)
        if len(sindp)==0: continue
        ftime=stime[sindp]
        depth=S.PRES_ADJUSTED.val[sindp]*1.01998
        temp=S.TEMP_ADJUSTED.val[sindp]
        salt=S.PSAL_ADJUSTED.val[sindp]
        alon=S.LONGITUDE.val[sindp]
        alat=S.LATITUDE.val[sindp]    
        for nn,ti in enumerate(ftime):
            if nn>0:
               if ftime[nn-1]==ti: continue
            sindp = near_pts(c_[alon[nn],alat[nn]], c_[C.SCHISM_hgrid_node_x.val, C.SCHISM_hgrid_node_y.val])
            sindt = abs(ctime - ti).argmin()
            sindp2 = near_pts(c_[alon[nn],alat[nn]], c_[D.SCHISM_hgrid_node_x.val, D.SCHISM_hgrid_node_y.val])
            if len(temp[nn,:].data[~temp[nn,:].mask])<3: continue
            if len(salt[nn,:].data[~salt[nn,:].mask])<3: continue
            #statistics
            # 1st model
            di=squeeze(-C2.zCoordinates.val[sindt,sindp,:]).data[~squeeze(-C2.zCoordinates.val[sindt,sindp,:]).mask]
            mtpi=squeeze(C3.temperature.val[sindt,sindp,:]).data[~squeeze(-C3.temperature.val[sindt,sindp,:]).mask]
            msi=squeeze(C4.salinity.val[sindt,sindp,:]).data[~squeeze(C4.salinity.val[sindt,sindp,:]).mask]
            omask=~temp[nn,:].mask*~salt[nn,:].mask*~depth[nn,:].mask
            odi=depth[nn,:].data[omask]
            otpi=temp[nn,:].data[omask]
            osi=salt[nn,:].data[omask]
            if odi.min()<di.min() or odi.max()>di.max():
               ofpt=(odi>=di.min())*(odi<=di.max())
            else:
               ofpt=full(len(odi),True)
            mtpii = interpolate.interp1d(flip(di),flip(mtpi))(odi[ofpt])
            msii = interpolate.interp1d(flip(di),flip(msi))(odi[ofpt])
            
            st_temp1=get_stat(mtpii,otpi[ofpt]);
            st_salt1=get_stat(msii,osi[ofpt]);

            # 2nd model
            di=squeeze(-D2.zCoordinates.val[sindt,sindp2,:]).data[~squeeze(-D2.zCoordinates.val[sindt,sindp2,:]).mask]
            mtpi=squeeze(D3.temperature.val[sindt,sindp2,:]).data[~squeeze(-D3.temperature.val[sindt,sindp2,:]).mask]
            msi=squeeze(D4.salinity.val[sindt,sindp2,:]).data[~squeeze(D4.salinity.val[sindt,sindp2,:]).mask]
            if odi.min()<di.min() or odi.max()>di.max():
               ofpt=(odi>=di.min())*(odi<=di.max())
            else:
               ofpt=full(len(odi),True)
            mtpii = interpolate.interp1d(flip(di),flip(mtpi))(odi[ofpt])
            msii = interpolate.interp1d(flip(di),flip(msi))(odi[ofpt])

            st_temp2=get_stat(mtpii,otpi[ofpt]);
            st_salt2=get_stat(msii,osi[ofpt]);
   
 
            #Plot
            figure(1,figsize=[7.2,3.5])
            clf()
            subplot(1,2,1)
            plot(px,py,'k')
            plot(alon,alat,'r+',markersize=20)
            xlabel('Longitude')
            ylabel('Latitude')
            subplot(1,2,2)
            plot(temp[nn,:],depth[nn,:],'r',lw=2)
            plot(squeeze(C3.temperature.val[sindt,sindp,:]),squeeze(-C2.zCoordinates.val[sindt,sindp,:]),'k',lw=2)
            plot(squeeze(D3.temperature.val[sindt,sindp2,:]),squeeze(-D2.zCoordinates.val[sindt,sindp2,:]),'b',lw=2)
            title('lon:{:.2f}, lat:{:.2f}, obs:{}, mdl:{}'.format(alon[nn],alat[nn],num2date(ti).strftime('%Y-%m-%d %H:%M:%S'),num2date(ctime[sindt]).strftime('%Y-%m-%d %H:%M:%S')), fontsize=11, fontweight='bold')
            xm=[temp[nn,:].data[~temp[nn,:].mask].min()-1,temp[nn,:].data[~temp[nn,:].mask].max()+1]; ym=[depth[nn,:].data[~depth[nn,:].mask].min(), depth[nn,:].data[~depth[nn,:].mask].max()]
            setp(gca(),ylim=ym,xlim=xm)
            gca().invert_yaxis()
            text(xm[0]+0.1*diff(xm),ym[0]+0.1*diff(ym),'mdl={}'.format(tags[0]),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.15*diff(ym),'R={:0.3f}'.format(st_temp1.R),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.2*diff(ym),'MAE={:0.3f}'.format(st_temp1.MAE),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.25*diff(ym),'ME={:0.3f}'.format(st_temp1.ME),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.3*diff(ym),'RMSE={:0.3f}'.format(st_temp1.RMSD),fontsize=12)

            text(xm[0]+0.1*diff(xm),ym[0]+0.5*diff(ym),'mdl={}'.format(tags[1]),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.55*diff(ym),'R={:0.3f}'.format(st_temp2.R),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.6*diff(ym),'MAE={:0.3f}'.format(st_temp2.MAE),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.65*diff(ym),'ME={:0.3f}'.format(st_temp2.ME),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.7*diff(ym),'RMSE={:0.3f}'.format(st_temp2.RMSD),fontsize=12)

            legend(['Obs',*tags])
            xlabel('Temperature ($^\circ$C)')
            ylabel('Depth (m)')
            savefig('{}/{}_tmp_{}'.format(sname,mti[fnn], num2date(ti).strftime('%Y%m%d%H%M%S')) + '.png')
            
            figure(1)
            clf()
            subplot(1,2,1)
            plot(px,py,'k')
            plot(alon,alat,'r+',markersize=20)
            xlabel('Longitude')
            ylabel('Latitude')
            subplot(1,2,2)
            plot(salt[nn,:],depth[nn,:],'r',lw=2)
            plot(squeeze(C4.salinity.val[sindt,sindp,:]),squeeze(-C2.zCoordinates.val[sindt,sindp,:]),'k',lw=2)
            plot(squeeze(D4.salinity.val[sindt,sindp2,:]),squeeze(-D2.zCoordinates.val[sindt,sindp2,:]),'b',lw=2)
            title('lon:{}, lat:{}, ARGO:{}, SCHISM:{}'.format(alon[nn],alat[nn],num2date(ti).strftime('%Y-%m-%d %H:%M:%S'),num2date(ctime[sindt]).strftime('%Y-%m-%d %H:%M:%S')), fontsize=11, fontweight='bold')
            xm=[salt[nn,:].data[~salt[nn,:].mask].min()-0.2,salt[nn,:].data[~salt[nn,:].mask].max()+0.2]; ym=[depth[nn,:].data[~depth[nn,:].mask].min(), depth[nn,:].data[~depth[nn,:].mask].max()]
            setp(gca(),ylim=ym,xlim=xm)
            gca().invert_yaxis()
            text(xm[0]+0.1*diff(xm),ym[0]+0.1*diff(ym),'mdl={}'.format(tags[0]),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.15*diff(ym),'R={:0.3f}'.format(st_salt1.R),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.2*diff(ym),'MAE={:0.3f}'.format(st_salt1.MAE),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.25*diff(ym),'ME={:0.3f}'.format(st_salt1.ME),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.3*diff(ym),'RMSE={:0.3f}'.format(st_salt1.RMSD),fontsize=12)

            text(xm[0]+0.1*diff(xm),ym[0]+0.5*diff(ym),'mdl={}'.format(tags[1]),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.55*diff(ym),'R={:0.3f}'.format(st_salt2.R),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.6*diff(ym),'MAE={:0.3f}'.format(st_salt2.MAE),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.65*diff(ym),'ME={:0.3f}'.format(st_salt2.ME),fontsize=12)
            text(xm[0]+0.1*diff(xm),ym[0]+0.7*diff(ym),'RMSE={:0.3f}'.format(st_salt2.RMSD),fontsize=12)

            legend(['Obs',*tags])
            xlabel('Salinity (PSU)')
            ylabel('Depth (m)')
            savefig('{}/{}_salt_{}'.format(sname,mti[fnn], num2date(ti).strftime('%Y%m%d%H%M%S')) + '.png')

print('done')
