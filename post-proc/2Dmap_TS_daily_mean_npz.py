from pylib import *
import cmocean
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs

gd=read_schism_hgrid('./04.gr3')
S=loadz('./RUN04a-TS-1_60.npz'); S.time=S.time+datenum(2020,1,1)
sname='./images/RUN04a/TS'
c1=[0, 30]
c2=[30,38]

# subset to exclude spin up
st=datenum('2020-1-15'); et=datenum('2020-3-1')
fpt=(st<=S.time)*(S.time<=et)
S.time=S.time[fpt]; S.temp=S.temp[fpt,:]; S.salt=S.salt[fpt,:]

# plot
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
title('Sea Surface Temperature ($^\circ$C)', fontsize=12, fontweight='bold')
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
title('Sea Surface Salinity (PSU)', fontsize=12, fontweight='bold')
ioff()

cbarlabels1 = linspace(c1[0],c1[1],9)
cbarlabels2 = linspace(c2[0],c2[1],9)
for nn,ctime in enumerate(arange(S.time[0],S.time[-1],1)):
    print('reading time= {}'.format(num2date(ctime).strftime('%Y-%m-%d')))
    
    ax1 = subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ssh=gd.plot(fmt=1,value=S.temp[nn*24:nn*24+23,:].mean(axis=0),cmap=cmocean.cm.thermal,clim=c1,cb=False)
    cb1=colorbar(ssh,orientation='horizontal',ticks=cbarlabels1)

    ax2 = subplot(1, 2, 2, projection=ccrs.PlateCarree())
    vel=gd.plot(fmt=1,value=S.salt[nn*24:nn*24+23,:].mean(axis=0),cmap=cmocean.cm.haline,clim=c2,cb=False)
    cb2=colorbar(vel,orientation='horizontal',ticks=cbarlabels2)

    suptitle(num2date(ctime).strftime('%Y-%m-%d'), fontsize=12, fontweight='bold')
#    tight_layout()
    savefig('{}/{}'.format(sname,num2date(ctime).strftime('%Y-%m-%d')) + '.png',bbox_inches='tight')
    for coll in ssh.collections: coll.remove()
    cb1.remove()
    for coll in vel.collections: coll.remove()
    cb2.remove()
    #close()
print('---------DONE---------')
