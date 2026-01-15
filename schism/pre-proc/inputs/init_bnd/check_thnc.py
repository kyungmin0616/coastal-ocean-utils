from pylib import *

C1=ReadNC('./uv3D.th.nc')
C2=ReadNC('./TEM_3D.th.nc')
C3=ReadNC('./SAL_3D.th.nc')
C4=ReadNC('./elev2D.th.nc')
levels=100
bdnno=arange(C4.dims[0])

figure(figsize=[22,9])
subplot(1, 5, 1)
contourf(C1.time.val/3600/24,bdnno, C1.time_series.val[:,:,-1,0].transpose(),levels=levels,cmap='jet')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('U')

subplot(1, 5, 2)
contourf(C1.time.val/3600/24,bdnno, C1.time_series.val[:,:,-1,1].transpose(),levels=levels,cmap='jet')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('V')

subplot(1, 5, 3)
contourf(C2.time.val/3600/24,bdnno, C2.time_series.val[:,:,-1,0].transpose(),levels=levels,cmap='jet')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('Temperature')

subplot(1, 5, 4)
contourf(C3.time.val/3600/24,bdnno, C3.time_series.val[:,:,-1,0].transpose(),levels=levels,cmap='jet')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('Salinity')

subplot(1, 5, 5)
contourf(C4.time.val/3600/24,bdnno, C4.time_series.val[:,:,0,0].transpose(),levels=levels,cmap='jet')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('SSH')
tight_layout(pad=2.0)

#show()
savefig('check_thnc.png',dpi=900,bbox_inches='tight')
#close()
