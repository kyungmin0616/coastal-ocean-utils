from pylib import *
import cmocean
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs

StartT=datenum(2013,1,1); EndT=datenum(2014,1,1)
dir_hycom='/rcfs/projects/mhk_modeling/dataset/CMEMS/NAO/reanalysis/'
sname='./images/CMEMS/ssh_vel'

c1=[-1.5, 1.5]
c2=[0,2]
levels = linspace(c1[0], c1[1], 101)
levels2 = linspace(c2[0], c2[1], 101)
#find all hycom files
fnames=array([i for i in os.listdir(dir_hycom) if i.endswith('.nc')])
mti=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

#extent=[gd.x.min()-1,gd.x.max()+1,gd.y.min()-1,gd.y.max()+1]
extent=[-98.8,-59,7.2,46.9]

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

for nn, fname in enumerate(fnames):
    S = ReadNC('{}/{}'.format(dir_hycom,fname))
    print('Reading {}'.format(fname))

    ax1 = subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ssh=contourf(S.longitude.val, S.latitude.val, squeeze(S.zos.val), levels=levels, cmap='nipy_spectral', extend='both')
    cb1=colorbar(orientation='vertical',fraction=0.05)

    ax2=subplot(1,2,2,projection=ccrs.PlateCarree())
    u=array(S.uo.val); v=array(S.vo.val); vel_meg=sqrt(u[0,0,:,:]**2+v[0,0,:,:]**2)
    fpt=vel_meg>10; vel_meg[fpt]=NaN
    vel=contourf(S.longitude.val, S.latitude.val, vel_meg, levels=levels2, cmap='jet', extend='both')
    cb2=colorbar(orientation='vertical',fraction=0.05)
    suptitle(num2date(mti[nn]).strftime('%Y-%m-%d'), fontsize=12, fontweight='bold')
    #tight_layout()
    savefig('{}/{}'.format(sname,num2date(mti[nn]).strftime('%Y-%m-%d')) + '.png',bbox_inches='tight')
    for coll in ssh.collections: coll.remove()
    cb1.remove()
    for coll in vel.collections: coll.remove()
    cb2.remove()
    #close()
print('---------DONE---------')
