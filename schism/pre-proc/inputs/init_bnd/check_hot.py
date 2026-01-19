from pylib import *
import matplotlib.pyplot as plt

gd=read_schism_hgrid('/S/data00/G6008/d1041/Projects/SendaiOnagawa/pre-proc/grid/01/hgrid.gr3')
hot='./hotstart.nc'


S=ReadNC('{}'.format(hot))

figure(1,figsize=[18,9])
subplot(1,5,1)
gd.plot(fmt=1,value=S.eta2.val, cmap='nipy_spectral',)
xlabel('Longitude'); ylabel('Latitude')
title('ssh')

subplot(1,5,2)
gd.plot(fmt=1,value=S.tr_nd.val[:,0,0], clim=[0,20],cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Temperature-bottom')

subplot(1,5,3)
gd.plot(fmt=1,value=S.tr_nd.val[:,-1,0],clim=[0,20], cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Temperature-top')

subplot(1,5,4)
gd.plot(fmt=1,value=S.tr_nd.val[:,0,1], clim=[0,36],cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Salinity-bottom')

subplot(1,5,5)
gd.plot(fmt=1,value=S.tr_nd.val[:,-1,1], clim=[0,36],cmap='jet',)
xlabel('Longitude'); ylabel('Latitude')
title('Salinity-top')

tight_layout()
show()
#savefig('check_hot.png',dpi=900,bbox_inches='tight')
#close()
