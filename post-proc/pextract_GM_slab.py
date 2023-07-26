from pylib import *
#from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
#import cartopy.crs as ccrs
#import cartopy.io.img_tiles as cimgt
#import cartopy.feature as cfeature


#dir_data='/rcfs/projects/mhk_modeling/dataset/HYCOM/FC'
dir_data='/rcfs/projects/mhk_modeling/dataset/CMEMS/NAO/reanalysis'
StartT,EndT=datenum(2013,1,1),datenum(2014,1,1)
sname=os.path.expanduser('./CMEMS-ssh-2D')
#bp=read_shapefile_data('NAO_bnd.shp')
#px,py=bp.xy.T

fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])
mti=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

C=zdata(); C.time=[]; C.elev=[]
for nn,fname in enumerate(fnames):
    print('{}'.format((nn+1)/len(fnames)*100))
    S=ReadNC('{}/{}'.format(dir_data,fname));
    C.time.append(mti[nn])
    #C.elev.append(S.surf_el.val[0])
    C.elev.append(S.zos.val[0])
C.lon=array(S.longitude.val); C.lat=array(S.latitude.val);C.time=array(C.time); C.elev=array(C.elev)

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

