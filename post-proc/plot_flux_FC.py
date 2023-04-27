from pylib import *
import pandas as pd

dir_obs='/rcfs/projects/mhk_modeling/dataset/NOAA/FloridaCurrent/FC_cable_transport_2016.dat'
StartT=[datenum('2016-9-8'),datenum('1950-1-1'),datenum('2000-1-1')]
st=datenum(2016,5,15);se=datenum(2016,11,1) #time window for plot
sst=datenum(2016,9,15);sse=datenum(2016,10,24) #time window for stat
dmean=1 # daily mean
csta=0 # statistical analysis
# plot control
ym=[15,40]
lw=[3,1,1,1];colors='kgbcm'; lstyle=['-','-','-','-','None','None','None','None']; markers=['None','None','None','None','*','^','o','None']

# model result
runs=['./flux/Paper_flux_2016.npz','./npz/CMEMS_flux_2016.npz','./npz/HYCOM_flux_2016.npz']
tags=['Paper','CMEMS','HYCOM']
##########################################################################################################################################################

#font size
SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 15

#rc('font', family='Helvetica')
rc('font', size=SMALL_SIZE)  # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#read obs results
obs=npz_data(); Data=loadtxt(dir_obs,skiprows=35); obs.fc=Data[:,3]
obs.time=[]
for n in arange(len(Data)):
    obs.time.append(datenum(str(Data[n,0].astype('int'))+'-'+str(Data[n,1].astype('int'))+'-'+str(Data[n,2].astype('int'))))
obs.time=array(obs.time); obs.fc=array(obs.fc)

#read model results
Model=[]
for m, run in enumerate(runs):
    if run.endswith('.npz'):
       if m==len(runs)-1: Si=loadz('{}'.format(run)); Si.time=Si.time/24+StartT[m]; Si.flux=squeeze(Si.flux)/1000000
       elif m==len(runs)-2: Si=loadz('{}'.format(run)); Si.time=Si.time/24+StartT[m]; Si.flux=squeeze(Si.flux)/1000000
       else: Si=loadz('{}'.format(run)); Si.time=Si.time+StartT[m]; Si.flux=-squeeze(Si.flux)/1000000; #Si.flux=lpfilt(Si.flux,1/36/24,13/24)
    elif run.endswith('.out'):
       Si=npz_data(); Data=loadtxt(run); Si.time=Data[:,0]+StartT[m]; Si.flux=Data[:,1:]/1000000
    else: print('Wrong data type')
    Model.append(Si)

#daily mean
if dmean==1:
    for m, run in enumerate(runs):
        times=array([num2date(i).strftime('%Y-%m-%d %H:%M:%S') for i in Model[m].time])
        times=pd.to_datetime(times); data=Model[m].flux
        df = pd.DataFrame(data=data,index=times); means=df.groupby(df.index.floor('D')).mean()
        times=array(datenum(means.index.astype(str))); mfc=means.values
        Model[m].time=times; Model[m].flux=mfc

#plot
figure(1,figsize=[18,9])
clf()
plot(obs.time,obs.fc,'r',lw=lw[0])
for n, run in enumerate(runs):
    if csta==1:
        mti=Model[n].time; myi=Model[n].flux
        fpt=(mti>=sst)*(mti<=sse); mti=mti[fpt]; myi=squeeze(myi[fpt])
        fpt = (obs.time >= mti.min()) * (obs.time <= mti.max()); oti = obs.time[fpt]; oyi = obs.fc[fpt]
        myii = interpolate.interp1d(mti, myi)(oti)
        stv=get_stat(myii,oyi); MEi=mean(myii)-mean(oyi)
    plot(Model[n].time,Model[n].flux,linestyle=lstyle[n],color=colors[n],marker=markers[n],lw=lw[n])
    if csta==1: text(st, (ylim()[1]+1.) + n*0.04 * diff(ylim()),'{}--> R: {:.2f}, RMSE: {:.2f}, ME: {:.2f}'.format(tags[n], stv.R, stv.RMSD, MEi), color='k') 
gca().xaxis.grid('on')
gca().yaxis.grid('on')
xts, xls = get_xtick(fmt=2, xts=arange(st, se+2,5), str='%m/%d')
setp(gca(), xticks=xts, xticklabels=xls, xlim=[st, se],ylim=ym)
legend(['NOAA cable',*tags])
xlabel('Date (2016)')
ylabel('Sv')
gcf().tight_layout()
show()
