from pylib import *
import pandas as pd
#from datetime import datetime

cl=[0,2.5]
ym=[40,300]

fname="B2_2013_Basic_NoQC_clean.txt"
bsize=5 #Vertical bin size (m)
vhight=12.57 #Vertical height above ADCP of middle of bin 1 (m)
tsize=32716
dsize=64

#fname="BIRENE_2011_Basic_NoQC_clean.txt"
#bsize=4 #Vertical bin size (m)
#vhight=12.06 #Vertical height above ADCP of middle of bin 1 (m)
#tsize=122889
#dsize=64

#fname="B2_2009_Basic_NoQC_clean.txt"
#bsize=5 #Vertical bin size (m)
#vhight=12.53 #Vertical height above ADCP of middle of bin 1 (m)
#tsize=18830
#dsize=67

### Loop the data lines
with open(fname, 'r') as temp_f:
    # get No of columns in each line
    col_count = [ len(l.split(",")) for l in temp_f.readlines() ]

### Generate column names  (names will be 0, 1, 2, ..., maximum columns - 1)
column_names = [i for i in range(0, max(col_count))]

### Read csv
df = pd.read_csv(fname, header=None, delimiter=",", names=column_names)
df=df.mask(df == ' ')

tidx=where(df.iloc[:,0]=='TIME[{}]'.format(tsize))[0][0]
didx=where(df.iloc[:,0]=='DEPTH[{}]'.format(tsize))[0][0]
dridx=where(df.iloc[:,0]=='DIR[{}x{}]'.format(tsize,dsize))[0][0]
vidx=where(df.iloc[:,0]=='VEL[{}x{}]'.format(tsize,dsize))[0][0]


atime=df.iloc[tidx+1:didx,0].values.astype(float); atime=datenum(pd.to_datetime(atime-719529,unit='D').values.astype('str'))
depth=df.iloc[didx+1:dridx,0].values.astype(float)
cdir=df.iloc[dridx+1:vidx,:dsize].values.astype(float)
cvel=df.iloc[vidx+1::,:dsize].values.astype(float); fpt=cvel==-32.768; cvel[fpt]=nan

# exclude deploment period
fpt=depth<depth.mean()/2;
atime=atime[~fpt]; depth=depth[~fpt]; cdir=cdir[~fpt,:]; cvel=cvel[~fpt,:] 

#mdepth=[]
#for i in arange(shape(cdir)[1]):
#    mdepth.append(bsize*(i+1))
#mdepth=flip(mdepth)
#tt,dd=meshgrid(atime,mdepth)
#tt=array(tt.transpose()); dd=array(dd.transpose())

#Measured depth
mdepth=[]; dd=[]
for j in arange(len(atime)):
    mdepth=[]
    for i in arange(shape(cdir)[1]):
        mdepth.append(depth[j]-vhight-bsize*i)
    dd.append(mdepth)
dd=array(dd)

# Time for 2d map
tt=[]
for i in arange(shape(cdir)[1]):
    tt.append(atime)
tt=array(transpose(tt))

### plot
# 2Dmap
levels=linspace(cl[0],cl[1],101)
figure(1,figsize=[17,5])
contourf(tt,dd,cvel,levels=levels,cmap='jet',extend='both');colorbar()
xts,xls = get_xtick(fmt=2, xts=[*linspace(atime.min(),atime.max(),10)], str='%d/%b/%y')
setp(gca(), xticks=xts, xticklabels=xls, xlim=[atime.min(),atime.max()],ylim=ym,yticks=linspace(ym[0],ym[1],7))
gca().invert_yaxis()
xlabel('Date'); ylabel('Depth (m)'); title('Current magnitude (m/s)')
show()

# vertical profile for min, mean and max
meanvel=nanmean(cvel,axis=0); maxvel=nanmax(cvel,axis=0);minvel=nanmin(cvel,axis=0);md=nanmean(dd,axis=0)
fpt=md>50; md=md[fpt]; meanvel=meanvel[fpt]; minvel=minvel[fpt]; maxvel=maxvel[fpt]
figure(2,figsize=[7,7])
plot(minvel,md,'b+')
plot(meanvel,md,'k+');
plot(maxvel,md,'r+')
ylim([0,350]);xlim([-0.5,2.5])
gca().invert_yaxis()
gca().xaxis.grid('on')
gca().yaxis.grid('on')
xlabel('Current magnitude (m/s)'); ylabel('Depth (m)'); title('Vertical profile')
show()
