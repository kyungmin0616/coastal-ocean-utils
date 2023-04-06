from pylib import *

gd2=read_schism_hgrid('./shapiro_ETOPO1.gr3')
gd=read_schism_hgrid('./shapiro.gr3')
#gd3=read_schism_hgrid('./shapiro_paper.gr3')


#gd3.x,gd3.y=proj_pts(gd3.x,gd3.y,'epsg:26918','epsg:4326')
gd2.x,gd2.y=proj_pts(gd2.x,gd2.y,'epsg:26918','epsg:4326')
gd.x,gd.y=proj_pts(gd.x,gd.y,'epsg:26918','epsg:4326')


figure(1, figsize=[18,9])
clf()
#subplot(1,2,1)
#gd.plot(fmt=1,cmap='jet')
gd.plot(fmt=1,value=gd.dp-gd2.dp,cmap='jet')

#subplot(1,2,2)
#gd2.plot(fmt=1,cmap='jet',clim=[gd.dp.min(),gd.dp.max()])
#setp(gca(),xlim=[gd.x.min(),gd.x.max()], ylim=[gd.y.min(),gd.y.max()])
#gd.plot(fmt=1,value=gd.dp-gd3.dp,cmap='jet')

#subplot(1,3,3)
#gd3.plot(fmt=1,cmap='jet',clim=[gd.dp.min(),gd.dp.max()])
#setp(gca(),xlim=[gd.x.min(),gd.x.max()], ylim=[gd.y.min(),gd.y.max()])
show()

