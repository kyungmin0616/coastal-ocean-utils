from pylib import *
import cmocean
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
import cartopy.crs as ccrs

StartT,EndT=datenum(2020,1,1),datenum(2020,2,1)
tres='2km' #1km,2km, 6km
sname='./images/2Dmap/HFRadar/2km'
dir_data='/rcfs/projects/mhk_modeling/dataset/HFRadar/www.ncei.noaa.gov/data/oceans/ndbc/hfradar/rtv/2020/202001/USEGC/'
c1=[0,2]
extent=[-90,-65,24,45]
levels = linspace(c1[0], c1[1], 101)

#filtering all files
fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])
res=[array(i.replace('.','_').split('_'))[3] for i in fnames]
res=array(res); fpt=res==tres; fnames=fnames[fpt]
mti=datenum([array(i.replace('.','_').split('_'))[0] for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

for nn, fname in enumerate(fnames):
    figure(1, figsize=[9, 7])
    clf()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_xticks(linspace(extent[0], extent[1], num=6), crs=ccrs.PlateCarree())
    ax.set_yticks(linspace(extent[2], extent[3], num=7), crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(number_format='.1f',zero_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.1f')
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    print('Reading {}'.format(fname))
    C=ReadNC('{}/{}'.format(dir_data,fname))
    contourf(C.lon.val,C.lat.val,sqrt(C.u.val[0,:,:]**2+C.v.val[0,:,:]**2),levels=levels,cmap='jet')
    colorbar(orientation='horizontal',fraction=0.05)
    savefig('{}/{}'.format(sname,num2date(mti[nn]).strftime('%Y%m%d%H%M%S')) + '.png')
