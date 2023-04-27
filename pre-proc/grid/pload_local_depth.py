from pylib import *

sname='./hgrid.ll.new.lcdepth'
grd='./hgrid.ll.new'
shpfile='./ETOPO_20M_50M.shp'
dems=['/people/park075/mhk_modeling/dataset/DEM/npz/ETOPO_2022_v1_15s_WNAO.npz','/people/park075/mhk_modeling/dataset/DEM/npz/CUDEM_SGA.npz']

gd=read_schism_hgrid(grd)
bp=read_shapefile_data(shpfile)
px,py=bp.xy.T

sindp=inside_polygon(c_[gd.x,gd.y],px,py)

fpt=sindp==1
ndp=gd.dp[fpt].copy()

for dem in dems:
    dpi,fpt2=load_bathymetry(gd.x[fpt],gd.y[fpt],'{}'.format(dem),fmt=1)
    ndp[fpt2]=-dpi

ndp2=gd.dp.copy()
ndp2[fpt]=ndp

figure(1,figsize=[12,8])
clf()
subplot(1,3,1)
gd.plot(fmt=1,cmap='jet')

subplot(1,3,2)
gd.plot(fmt=1,value=ndp2,cmap='jet')

subplot(1,3,3)
gd.plot(fmt=1,value=ndp2-gd.dp,cmap='jet')
gcf().tight_layout()
#show()
savefig('./local_bathy.png')
close()


gd.dp=ndp2
gd.write_hgrid(fmt=0,fname=sname)
proj(sname,0,'epsg:4326','./hgrid.utm',0,'epsg:26918')
