from pylib import *
import cmocean
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs

gd=read_schism_hgrid('./08.gr3')
sname='./images/RUN08g/ssh_vel'
run='../run/RUN08g/outputs'
stacks=[1,360]
istacks=[*arange(stacks[0],stacks[1]+1)]
refdate='2013-01-01'
c1=[-1.5, 1.5]
c2=[0,2]

extent=[gd.x.min()-1,gd.x.max()+1,gd.y.min()-1,gd.y.max()+1]

figure(1, figsize=[18, 7])
clf()
ax = subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
ax.set_xticks(linspace(extent[0], extent[1], num=6), crs=ccrs.PlateCarree())
ax.set_yticks(linspace(extent[2], extent[3], num=7), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(number_format='.1f', zero_direction_label=True)
lat_formatter = LatitudeFormatter(number_format='.1f')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
title('Sea Surface Height (m)', fontsize=12, fontweight='bold')
ioff()
ax2 = subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax2.set_extent(extent, crs=ccrs.PlateCarree())
ax2.coastlines()
ax2.set_xticks(linspace(extent[0], extent[1], num=6), crs=ccrs.PlateCarree())
ax2.set_yticks(linspace(extent[2], extent[3], num=7), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(number_format='.1f', zero_direction_label=True)
lat_formatter = LatitudeFormatter(number_format='.1f')
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
title('Surface current (m/s)', fontsize=12, fontweight='bold')
ioff()

for istack in istacks:
    print('reading stack= {}'.format(istack))
    C=ReadNC('{}/out2d_{}.nc'.format(run,istack))
    ctime = array(C.time.val)/86400+datenum(refdate);
    elev=array(C.elevation.val)
    mask=array(C.dryFlagNode.val)
    fpt=mask==1
    elev[fpt]=NaN
    ax1 = subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ssh=gd.plot(fmt=1,value=elev.mean(axis=0),cmap='nipy_spectral',clim=c1,cb=False)
    cb1=colorbar(ssh,orientation='vertical',fraction=0.05)

    C=ReadNC('{}/horizontalVelX_{}.nc'.format(run,istack))
    C2=ReadNC('{}/horizontalVelY_{}.nc'.format(run,istack))
    u=array(C.horizontalVelX.val); v=array(C2.horizontalVelY.val); vel_meg=sqrt(u[:,:,-1]**2+v[:,:,-1]**2)
    vel_meg[fpt]=NaN
    ax2 = subplot(1, 2, 2, projection=ccrs.PlateCarree())
    vel=gd.plot(fmt=1,value=vel_meg.mean(axis=0),cmap='jet',clim=c2,cb=False)
    cb2=colorbar(vel,orientation='vertical',fraction=0.05)
    suptitle(num2date(ctime[0]).strftime('%Y-%m-%d'), fontsize=12, fontweight='bold')
#    tight_layout()
    savefig('{}/{}'.format(sname,num2date(ctime[0]).strftime('%Y-%m-%d')) + '.png',bbox_inches='tight')
    for coll in ssh.collections: coll.remove()
    cb1.remove()
    for coll in vel.collections: coll.remove()
    cb2.remove()
    #close()
print('---------DONE---------')
