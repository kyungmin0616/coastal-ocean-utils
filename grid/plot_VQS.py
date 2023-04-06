from pylib import *

z_m=loadtxt('vgrid_master.out') # m,nv(m),hsm,z_mas()
z1=loadtxt('transect1.out') # i,kbp,x,y,transect_dis,dp,set_flag,z-coor
ms=15

np_m=shape(z_m)[0]
nv_m=shape(z_m)[1]-3
zcor_m=z_m[:,3::].copy()

kbp_m=[]
for i in arange(np_m):
    kbp_m=z_m[i,1]; 
    zcor_m[i,kbp_m.astype('int')::]=NaN

np=shape(z1)[0]
nvrt=shape(z1)[1]-7
zcor1=z1[:,7::]

figure(1,figsize=[17,9])
subplot(2,1,1)
plot(z_m[:,0],zcor_m,'k-')
plot(z_m[:,0],-z_m[:,2],'r.',ms=ms);
for i in arange(np_m):
    plot(z_m[i,0]*ones(nv_m),zcor_m[i,:],'k')
title('Master grid')
xlabel('Grid #'); ylabel('Depth (m)')



subplot(2,1,2)
plot(z1[0,4]*ones(len(zcor1[0,:])),zcor1[0,:],'k-')
plot(z1[0,4],-z1[0,5],'r.',ms=ms)
start=0

for i in arange(np):
    if i>0:
        if abs(z1[i,6]-z1[i-1,6])>1.e-3:
            plot(z1[start,i-1,4], zcor1[start:i-1,:],'k-')
            plot(z1[start:i-1,4], -z1[start:i-1,5],'r.',ms=ms)
    plot(z1[i,4]*ones(nvrt),zcor1[i,:],'k')

#plot last seg
plot(z1[start::,4],zcor1[start::,:],'k')
plot(z1[start::,4],-z1[start::,5],'r.',ms=ms)

title('Transect before adjustment (transect1)');
xlabel('Along transect distance (m)'); ylabel('Depth (m)')   

gcf().tight_layout()
show()
