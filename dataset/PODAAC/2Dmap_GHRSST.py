from pylib import *
import cmocean
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

sname='./images/'
dir_sst='./2016/'
xl=[-105, -60];yl=[8, 46]
levels=linspace(15,32,50)

# sorting files
StartT=datenum(2016,10,4); EndT=datenum(2016,11,30);
fnames=array([i for i in os.listdir(dir_sst) if i.endswith('.nc')])
mti=datenum(array([array(i.replace('.','-').split('-')[0]) for i in fnames]))
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

C=ReadNC('{}/{}'.format(dir_sst,fnames[0]))
sindpx=(C.lon.val>xl[0])*(C.lon.val<xl[1])
sindpy=(C.lat.val>yl[0])*(C.lat.val<yl[1])

tiles = cimgt.GoogleTiles(style = 'satellite')
tiles_res = 6
extent = [-105, -60, 8, 46] # ECGOM

figure(1, figsize=[9, 9])
clf()
ax = plt.axes(projection=tiles.crs)
ax.set_extent(extent, crs=ccrs.PlateCarree());
ax.add_image(tiles, tiles_res,interpolation='spline36')
ax.set_xticks(arange(extent[0],extent[1],abs(extent[0]-extent[1])/6), crs=ccrs.PlateCarree())
ax.set_yticks(arange(extent[2],extent[3],abs(extent[2]-extent[3])/6), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(number_format='.2f',zero_direction_label=True)
lat_formatter = LatitudeFormatter(number_format='.2f')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
xlabel('Longitude'); ylabel('Latitude')
ioff()

for n,fname in enumerate(fnames):
    C=ReadNC('{}/{}'.format(dir_sst,fname))
    tmp=C.analysed_sst.val[0,sindpy,:]
    tmp=tmp[:,sindpx]
    ctime=C.time.val.data/86400+datenum('1981-01-01 00:00:00')
    gz=contourf(C.lon.val[sindpx], C.lat.val[sindpy], tmp-273.15,levels=levels,transform=ccrs.PlateCarree(),cmap=cmocean.cm.thermal,extend='both')
    title(num2date(ctime[0]), fontsize=16, fontweight='bold')
    #colorbar()
    savefig('{}/{}'.format(sname,num2date(ctime[0]).strftime('%Y%m%d')) + '.png')
    for coll in gz.collections: coll.remove()

print('done')
