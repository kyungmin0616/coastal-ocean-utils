from pylib import *
#from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
#import cartopy.crs as ccrs
#import cartopy.io.img_tiles as cimgt
#import cartopy.feature as cfeature


dir_data='/rcfs/projects/mhk_modeling/dataset/HYCOM/FC'
StartT,EndT=datenum(2013,1,1),datenum(2014,1,1)
sname=os.path.expanduser('./HYCOM-elev-FC')
bp2=read_schism_bpfile('transect.bp')
#bp=read_shapefile_data('NAO_bnd.shp')
#px,py=bp.xy.T

fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])
mti=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

S=ReadNC('{}/{}'.format(dir_data,fnames[0]));
lonidx=[]; latidx=[]
for i in arange(bp2.nsta):
    lonidx.append(argmin(abs(S.lon.val-bp2.x[i])))
    latidx.append(argmin(abs(S.lat.val-bp2.y[i])))
lonidx=array(lonidx); latidx=array(latidx)
[val, fidx]=unique(lonidx, return_index=True); lonidx=lonidx[fidx]; latidx=latidx[fidx]
lon=S.lon.val[lonidx]; lat=S.lat.val[latidx]

C=zdata(); C.lon=lon; C.lat=lat; C.time=[]; C.elev=[]
for nn,fname in enumerate(fnames):
    print('{}'.format((nn+1)/len(fnames)*100))
    S=ReadNC('{}/{}'.format(dir_data,fname));
    C.time.append(mti[nn])
    C.elev.append(S.surf_el.val[0][latidx,lonidx])
C.lon=array(C.lon); C.lat=array(C.lat);C.time=array(C.time); C.elev=array(C.elev)

savez(sname,C)

#sname='images/GM/HYCOM/elev_along_FC/'
# plot profile of the elev along the FC
#for nn,ctime in enumerate(S.time):
#    figure(1,figsize=[8,5])
#    clf()
#    plot(S.lon,S.elev[nn],'k',lw=3)
#    xlim(xl); ylim(yl)
#    title(num2date(ctime))
#    savefig('{}/{}'.format(sname,num2date(ctime).strftime('%Y%m%d%H%M%S')) + '.png',bbox_inches='tight')
#    close()

