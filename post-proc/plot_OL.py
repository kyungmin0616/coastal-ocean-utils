from pylib import *
import pandas as pd
#from datetime import datetime

datetime.datetime(2013, 1, 1) + datetime.timedelta(4.2480 - 1)

cl=[0,2.5]
cl2=[10,33]
ym=[0,800]

fname='/rcfs/projects/mhk_modeling/dataset/OleanderLine/2013/OL_136613.asc'
year=2013
data=loadtxt(fname,skiprows=1)

atime=[]
for i in arange(shape(data)[0]):
    atime.append(datenum((datetime.datetime(year, 1, 1) + datetime.timedelta(data[i,0])).strftime("%y-%m-%d %H:%M:%S")))
atime=array(atime)

lon=data[:,1]; lat=data[:,2]; depth=data[:,3]; u=data[:,4]; v=data[:,5]; temp=data[:,9]; del data

###Plot
# 2D map for current (time vs depth)
levels=linspace(cl[0],cl[1],101)
figure(1,figsize=[17,5])
clf()
#tricontourf(atime,depth,sqrt(u**2+v**2),levels=levels,cmap='jet',extend='both');colorbar()
scatter(atime,depth,s=20,c=sqrt(u**2+v**2), marker = 'o',vmin=0,vmax=2, cmap = 'jet' );
xts,xls = get_xtick(fmt=2, xts=[*linspace(atime.min(),atime.max(),10)], str='%d/%b/%y')
setp(gca(), xticks=xts, xticklabels=xls, xlim=[atime.min(),atime.max()],ylim=ym,yticks=linspace(ym[0],ym[1],7))
gca().invert_yaxis()
xlabel('Date'); ylabel('Depth (m)'); title('Current magnitude (m/s)')
show()

# 2D map for current (lon vs depth)
levels=linspace(cl[0],cl[1],101)
figure(1,figsize=[17,5])
clf()
#tricontourf(atime,depth,sqrt(u**2+v**2),levels=levels,cmap='jet',extend='both');colorbar()
scatter(lon-360,depth,s=20,c=sqrt(u**2+v**2), marker = 'o',vmin=0,vmax=2, cmap = 'jet' );
setp(gca(),ylim=ym,yticks=linspace(ym[0],ym[1],7))
gca().invert_yaxis()
xlabel('longitude'); ylabel('Depth (m)'); title('Current magnitude (m/s)')
show()

# 2D map for temp (lon vs depth)
figure(1,figsize=[17,5])
clf()
#tricontourf(atime,depth,sqrt(u**2+v**2),levels=levels,cmap='jet',extend='both');colorbar()
scatter(lon-360,depth,s=20,c=temp, marker = 'o',vmin=cl2[0],vmax=cl2[1], cmap = 'jet' );
setp(gca(),ylim=ym,yticks=linspace(ym[0],ym[1],7))
gca().invert_yaxis()
xlabel('longitude'); ylabel('Depth (m)'); title('Temperature (deg)')
show()
