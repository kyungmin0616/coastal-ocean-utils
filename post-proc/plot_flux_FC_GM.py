from pylib import *
import pandas as pd

dir_obs='/rcfs/projects/mhk_modeling/dataset/NOAA/FloridaCurrent/FC_cable_transport_2016.dat'
StartT=[datenum('2016-9-8'),datenum(1950,1,1),datenum(2000,1,1)]
st=datenum(2016,1,1); se=datenum(2016,10,21)
runs=['npz/CMEMS_flux.npz','npz/HYCOM_flux_2016.npz']
ym=[0,40.0]
colors='kgbcm'

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

#read model results
Model=[]
for m, run in enumerate(runs):
    if run.endswith('.npz'):
       if m==len(runs)-1: Si=loadz('{}'.format(run)); Si.time=Si.time/24+StartT[m]; Si.flux=-squeeze(Si.flux)/1000000
       elif m==len(runs)-2: Si=loadz('{}'.format(run)); Si.time=Si.time/24+StartT[m]; Si.flux=-squeeze(Si.flux)/1000000
       else: Si=loadz('{}'.format(run)); Si.time=Si.time+StartT[m]; Si.flux=-squeeze(Si.flux)/1000000
    elif run.endswith('.out'):
       Si=npz_data(); Data=loadtxt(run); Si.time=Data[:,0]+StartT[m]; Si.flux=Data[:,1:]/1000000
    else: print('Wrong data type')
    Model.append(Si)

#daily mean
for m, run in enumerate(runs):
    times=array([num2date(i).strftime('%Y-%m-%d %H:%M:%S') for i in Model[m].time])
    times=pd.to_datetime(times); data=Model[m].flux
    df = pd.DataFrame(data=data,index=times); means=df.groupby(df.index.floor('D')).mean()
    times=array(datenum(means.index.astype(str))); mfc=means.values
    Model[m].time=times; Model[m].flux=mfc


#plot
figure(1,figsize=[18,9])
clf()
plot(obs.time,obs.fc,'r',lw=3)
for n, run in enumerate(runs):
    mti=Model[n].time; myi=Model[n].flux
    fpt=(mti>=sst)*(mti<=sse); mti=mti[fpt]; myi=squeeze(myi[fpt])
    fpt = (obs.time >= mti.min()) * (obs.time <= mti.max()); obs.time = obs.time[fpt]; obs.fc = obs.fc[fpt]
    myii = interpolate.interp1d(mti, myi)(obs.time)
    stv=get_stat(myii,obs.fc); MEi=mean(myii)-mean(obs.fc)
    plot(Model[n].time,Model[n].flux,linestyle=lstyle[n],color=colors[n],marker=markers[n],lw=lw[n])
    text(st, (ylim()[1]+1.) + n*0.04 * diff(ylim()),'{}--> R: {:.2f}, RMSE: {:.2f}, ME: {:.2f}'.format(tags[n], stv.R, stv.RMSD, MEi), color='k')

gca().xaxis.grid('on')
gca().yaxis.grid('on')
xts, xls = get_xtick(fmt=2, xts=arange(st, se+15,15), str='%m/%d')
setp(gca(), xticks=xts, xticklabels=xls, xlim=[st, se],ylim=ym)
legend(['NOAA cable',*runs])
xlabel('Date (2020)')
ylabel('Sv')
show()
