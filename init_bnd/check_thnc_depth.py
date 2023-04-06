from pylib import *

sname='./images_thnc/CMEMS/'

C1=ReadNC('./uv3D.th.nc')
C2=ReadNC('./TEM_3D.th.nc')
C3=ReadNC('./SAL_3D.th.nc')
C4=ReadNC('./elev2D_CMEMS.th.nc')

bdnno=arange(C4.dims[0])

for i in arange(C4.dims[1]):

    figure(figsize=[22,9])
    subplot(1, 4, 1)
    contourf(C1.time.val/3600/24,bdnno, C1.time_series.val[:,:,i,0].transpose(),cmap='jet')
    colorbar()
    xlabel('days')
    ylabel('Boundary no.')
    title('U')

    subplot(1, 4, 2)
    contourf(C1.time.val/3600/24,bdnno, C1.time_series.val[:,:,i,1].transpose(),cmap='jet')
    colorbar()
    xlabel('days')
    ylabel('Boundary no.')
    title('V')

    subplot(1, 4, 3)
    contourf(C2.time.val/3600/24,bdnno, C2.time_series.val[:,:,i,0].transpose(),cmap='jet')
    colorbar()
    xlabel('days')
    ylabel('Boundary no.')
    title('Temperature')

    subplot(1, 4, 4)
    contourf(C3.time.val/3600/24,bdnno, C3.time_series.val[:,:,i,0].transpose(),cmap='jet')
    colorbar()
    xlabel('days')
    ylabel('Boundary no.')
    title('Salinity')
    gcf().tight_layout()
    savefig('{}/TSUV_{}'.format(sname,i) + '.png')
    close()

figure(1,figsize[9,9])
clf()
contourf(C4.time.val/3600/24,bdnno, C4.time_series.val[:,:,0,0].transpose(),cmap='jet')
colorbar()
xlabel('days')
ylabel('Boundary no.')
title('SSH')
gcf().tight_layout()
avefig('{}/elev'.format(sname) + '.png')
close()

