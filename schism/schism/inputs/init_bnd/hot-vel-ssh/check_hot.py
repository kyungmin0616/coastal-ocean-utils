from pylib import *

gd=read_schism_hgrid('./hgrid.gr3')
gd.compute_all()
hot='./cmems_hot.nc'


S=ReadNC('{}'.format(hot))

figure(1,figsize=[18,4.5])
subplot(1,5,1)
gd.plot(fmt=1,value=S.eta2.val, cmap='nipy_spectral',)
xlabel('Longitude'); ylabel('Latitude')
title('ssh')

subplot(1,5,2)
gd.plot(fmt=1,value=S.tr_nd.val[:,0,0], cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Temperature-bottom')

subplot(1,5,3)
gd.plot(fmt=1,value=S.tr_nd.val[:,-1,0], cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Temperature-top')

subplot(1,5,4)
gd.plot(fmt=1,value=S.tr_nd.val[:,0,1], cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Salinity-bottom')

subplot(1,5,5)
gd.plot(fmt=1,value=S.tr_nd.val[:,-1,1], cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Salinity-top')

tight_layout()
show()

figure(2,figsize=[12,9])
subplot(1,4,1)
tricontourf(gd.xcj,gd.ycj,S.su2.val[:,1],levels=51, cmap='jet')
xlabel('Longitude'); ylabel('Latitude')
title('u-bottom')

subplot(1,4,2)
tricontourf(gd.xcj,gd.ycj,S.su2.val[:,-1],levels=51, cmap='jet')
xlabel('Longitude'); ylabel('Latitude')
title('u-top')

subplot(1,4,3)
tricontourf(gd.xcj,gd.ycj,S.sv2.val[:,1],levels=51, cmap='jet')
xlabel('Longitude'); ylabel('Latitude')
title('v-bottom')

subplot(1,4,4)
tricontourf(gd.xcj,gd.ycj,S.sv2.val[:,-1],levels=51, cmap='jet')
xlabel('Longitude'); ylabel('Latitude')
title('v-top')

tight_layout()
show()
#savefig('check_hot.png',dpi=900,bbox_inches='tight')
#close()
