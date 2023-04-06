from pylib import *

gd=read_schism_hgrid('../../../grid/04/hgrid.gr3')
hot='./hotstart.nc'


S=ReadNC('{}'.format(hot))

figure(1,figsize=[18,9])
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
#savefig('check_hot.png',dpi=900,bbox_inches='tight')
#close()
