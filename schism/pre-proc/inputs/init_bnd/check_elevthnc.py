from pylib import *

C4=ReadNC('./elev2D.aviso.th.nc')
C3=ReadNC('./elev2D.cmems.th.nc')

#levels=linspace(C4.time_series.val[:,:,0,0].min(),C4.time_series.val[:,:,0,0].max(),100)
levels=100
bdnno=arange(C4.dims[0])
figure(1,figsize=[18,9])
subplot(1,3,1)
contourf(C4.time.val/3600/24,bdnno, C4.time_series.val[:,:,0,0].transpose(),levels=levels,cmap='jet',extend='both')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('AVISO')
tight_layout(pad=2.0)

subplot(1,3,2)
contourf(C3.time.val/3600/24,bdnno, C3.time_series.val[:,:,0,0].transpose(),levels=levels,cmap='jet',extend='both')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('CMEMS')
tight_layout(pad=2.0)

subplot(1,3,3)
contourf(C4.time.val/3600/24,bdnno, C4.time_series.val[:,:,0,0].transpose()-C3.time_series.val[:,:,0,0].transpose(),levels=levels,cmap='jet')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('Diff.')
tight_layout(pad=2.0)

show()

