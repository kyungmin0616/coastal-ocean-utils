from pylib import *
import pandas as pd
import numpy as np

usgs=['USGS_02231254_RD.csv']

st=datenum(2021,12,20)
et=datenum(2022,3,30)

ntime=arange(0,(et-st)*86400,900)
sname='flux.th_2022_kp'
newset=ntime.copy()


for file in usgs:

    df = pd.read_csv(file,skiprows=26)
    df = df.drop([0])
    if df['tz_cd'][1]=='EDT': print('{} has EDT'.format(file));df['tz_cd'][1]='Etc/GMT+4'
    time=pd.to_datetime(df['datetime']).dt.tz_localize(df['tz_cd'][1]).dt.tz_convert('GMT')
    time= datenum(time.values.astype('str')).astype('float')
    rd=df[df.columns[4]].values.astype('float')*0.0283

    #cut target time
    fpt=(time>=st)*(time<=et); time=time[fpt]; rd=rd[fpt]

    #manage time
    time=(time-st)*86400; time,idx=unique(time,return_index=True); rd=rd[idx]
    nrd = -interpolate.interp1d(time, rd)(ntime)
    newset=column_stack((newset,nrd))

    #plot for checking data
    #plot(time,rd)

#

# save flux.th
np.savetxt('{}'.format(sname),newset,fmt='%f')

#check flux.th

#fs=loadtxt('flux.th_2022');
#fs2=loadtxt('flux.th_2022_kp')


#for nn in arange(shape(fs)[1]-1):
#    plot(fs[:,0],fs[:,nn+1],'r')
#    plot(fs2[:,0],fs2[:,nn+1],'b')

#show()
