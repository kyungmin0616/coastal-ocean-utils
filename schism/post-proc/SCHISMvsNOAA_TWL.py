#!/usr/bin/env python3
from pylib import *
from matplotlib.backends.backend_pdf import PdfPages
close("all")
#------------------------------------------------------------------------------
#input
# ------------------------------------------------------------------------------
StartT_model=[datenum(2016,9,8),datenum(2016,9,8)] #plot start time, model start time

chnpz=0
chnr=0
runs=['RUN12a-elev-w-tide-xyz.npz','RUN13a-elev-w-tide-xyz.npz']
oskp = 10

tags=['RUN12a','RUN13a'];
bpfile='./station_ver2.in'
stt=datenum('2016-9-25')
edt=datenum('2016-10-25')
regions=['None']

#figure save names; not save if it is None
#sname=None
# sname='/Users/park075/Dropbox (GaTech)/GaTech/Journal/2nd_GulfStream/Figure/Supple/Fig_3' # name for savefig
#sname=os.path.expanduser('./images/wl/RUN12a/TWL')
sname=os.path.expanduser('./images/wl/RUN12a-13a/') #images/wl/RUN12a-outnpz
#linestypes for different runs
colors='kgcm'; lstyle=['-','-','-','-','-']; markers=['None','None','None','None','None']
# colors='gbc'; lstyle=['-','-','-']; markers=['None','.','None']

#Axis limits
ym=[-1.5,2.7]; # SAB
# ym=[-1.,1.]; # MAB

##################################  read obs data
C1 = loadz('NOAA/Matthew/WL/noaa_elev_navd.npz')
C = loadz('NOAA/Matthew/WL/noaa_elev_msl.npz')
C3=loadz('NOAA/Matthew/TIDE/noaa_elev_navd.npz')
C2=loadz('NOAA/Matthew/TIDE/noaa_elev_msl.npz')


################################## stations to plots
noaa_stations_groups = {

'FT': array([11]),

'SAB': array([7,8,10,11,12,13,15,16]), # 16 ,17 need to be included in KMP

'MAB': array([19,21,23,26,27,28,30]),

'GOME': array([33,36,37,38]),

'GOMX': array([44, 49, 57, 59, 64, 67, 74, 77, 78, 2, 3]), #44~64 west; 67~3 east

'BPT': array([40,151,152,153,155,161] ),

'SABMAB': array([7,8,10,11,12,13,15,16,19,21,23,26,27,28,30] ),

'ECOAST': array([10,11,12,13,15,19,21,23,26,27,28,30,33,36,37,38] ),

'ECOAST-GOMX': array([7,8,10,11,12,13,15,19,21,23,26,27,28,30,33,36,37,38,44, 49, 57, 59, 64, 67, 74, 77, 78, 2, 3] ),

'All': array([2, 7,8,10,11,12,13,15,19,21,23,26,27,28,30,33,36,37,38,3, 78, 74, 64, 59, 49, 44, 40, 151,152,153,155,161,57,77,67])

}

for region in regions:
    if region == 'None': stations=None
    else: stations=noaa_stations_groups[region]

    #------------------------------------------------------------------------------
    #read station data
    #------------------------------------------------------------------------------
    #read station.bp
    fid=open('./stanames.txt'); staname=dict(zip(*array([i.strip().split(',') for i in fid.readlines()]).T)); fid.close()
    bp=read_schism_bpfile(bpfile);  bp.staname=array([staname[i] for i in bp.station])

    #for subset of stations
    if stations is not None:
        staid=array(stations).astype('int')-1
        bp.nsta=len(stations); bp.x=bp.x[staid]; bp.y=bp.y[staid]; bp.z=bp.z[staid]; bp.station=bp.station[staid]; bp.staname=bp.staname[staid]
    else:
        staid=arange(bp.nsta).astype('int')
        stations=arange(bp.nsta)+1

    ##################################  read model results
    Model=[]
    for m, run in enumerate(runs):
        if run.endswith('out_1'):
            Si = npz_data();
            Data = loadtxt(run);
            Si.time = Data[:, 0] / 86400 + StartT_model[m];
            Si.dth = (Data[:, 0][1] - Data[:, 0][0]) / 3600
            Si.elev = Data[:, 1:].transpose()
        elif run.endswith('.npz'):
            Si=loadz('{}'.format(run))
            Si.time=Si.time+StartT_model[m]; Si.elev=Si.elev
            Si.dth=(Si.time[1]-Si.time[0])*24
        else: print('Wrong data type')
        Model.append(Si)
    del Si



# -----------------------plot---------------------------------------------------
    maxfig=math.ceil(len(stations)/2)*2
    nosubplot=12
    nosubplotcl=3
    cntsta=0
    for m in arange(round(maxfig/nosubplot)):
        i1=m*nosubplot; i2=i1+nosubplot-1; iflag=0
        if i2 >= bp.nsta: i2=bp.nsta-1
        MAE=[]; Corr=[]; ME=[]; RMSE=[];
        # figure(m, figsize=[18, 9])
        # clf()
        for i in arange(i1,i2):
            figure(1,figsize=[15,4])
            clf()
            station=int(bp.station[i])
            # subplot(int(nosubplot/nosubplotcl),nosubplotcl,iflag)
            if bp.x[i]<= -81.7 and bp.y[i] <= 31.01: regname='GOMX'; #plot(bp.x[i],bp.y[i],'r.')
            elif bp.x[i] > -81.7 and bp.x[i] <=-70 and bp.y[i] <= 35.4: regname='SAB'; #plot(bp.x[i],bp.y[i],'b.')
            elif bp.x[i] > -78 and bp.x[i] <=-68 and bp.y[i] > 35.4 and bp.y[i] <=42.041: regname='MAB'; #plot(bp.x[i],bp.y[i],'g.')
            elif bp.x[i] > -71.4 and bp.x[i] <=-62 and bp.y[i] > 42.041 and bp.y[i] <=46.071: regname='GOME'; #plot(bp.x[i],bp.y[i],'k.')
            elif bp.x[i] > -65.5 and bp.x[i] <=-63.5 and bp.y[i] > 31.5 and bp.y[i] <=33: regname='BM'; #plot(bp.x[i],bp.y[i],'m.')
            else: regname='PT'; #plot(bp.x[i],bp.y[i],'y.')

            #find target time and station of obs
            lobs='None'
            fp=(C.station==station)*(C.time>stt)*(C.time<edt); oti=C.time[fp]; oyi=C.elev[fp]; lobs='msl'
            fp = (C2.station == station) * (C2.time > stt) * (C2.time < edt); oti_tide = C2.time[fp]; oyi_tide = C2.elev[fp];

            if sum(fp)==0:
                fp=(C1.station==station)*(C1.time>stt)*(C1.time<edt); oti=C1.time[fp]; oyi=C1.elev[fp]; lobs='navd'
                fp = (C3.station == station) * (C3.time > stt) * (C3.time < edt); oti_tide = C3.time[fp]; oyi_tide = C3.elev[fp];

            #add nan data between oti
            if len(oti)>100:
               ts=find_continuous_sections(oti,1.0); eoti=array([i[-1]+1/24 for i in ts.sections]); eoyi=ones(len(eoti))*nan
               oti=r_[oti,eoti]; oyi=r_[oyi,eoyi]; sind=argsort(oti); oti=oti[sind]; oyi=oyi[sind]
               ts=find_continuous_sections(oti_tide,1.0); eoti=array([i[-1]+1/24 for i in ts.sections]); eoyi=ones(len(eoti))*nan
               oti_tide=r_[oti_tide,eoti]; oyi_tide=r_[oyi_tide,eoyi]; sind=argsort(oti_tide); oti_tide=oti_tide[sind]; oyi_tide=oyi_tide[sind]
            if len(oyi)==0: continue
            if chnr==1:
                 if len(oyi)==len(oyi_tide):
                     foyi=oyi-oyi_tide
                     fpn = ~isnan(oyi);
                     oti = oti[fpn];
                     oyi = oyi[fpn]
                     foyi = lpfilt(oyi, 1 / 240, 13/ 24)
                     fptc = (oti > datenum('2016-09-18')) * (oti < datenum('2016-10-4'))
                     plot(oti, foyi - nanmean(foyi[fptc]), 'r', lw=2)
                 else:
                     foyi = lpfilt(oyi, 1 / 240, 13/ 24)
                     fptc = (oti > datenum('2016-09-18')) * (oti < datenum('2016-10-4'))
                     plot(oti, foyi - nanmean(foyi[fptc]), 'r', lw=2)
            else:
                 plot(oti, oyi, 'r', lw=1.)
                 # plot(oti[::oskp], oyi[::oskp], 'r.', ms=5, markerfacecolor="None", lw=1.)

            # plots and stats for models
            for nn, run in enumerate(runs):
                mti = Model[nn].time
                myi = Model[nn].elev[i, :];

                fpn = ~isnan(oyi); oti = oti[fpn]; oyi = oyi[fpn]
                fpt = (oti >= mti.min()) * (oti <= mti.max());
                otii = oti[fpt]; oyii = oyi[fpt]
                myii = interpolate.interp1d(mti, myi)(otii)

                st=get_stat(myii,oyii); MEi=mean(myii)-mean(oyii)
                RMSE.append(st.RMSD); MAE.append(st.MAE); Corr.append(st.R); ME.append(MEi)

                plot(mti,myi-MEi,linestyle=lstyle[nn],color=colors[nn],marker=markers[nn],ms=3,alpha=0.85,lw=1)
                xts, xls = get_xtick(fmt=2, xts=linspace(stt,edt,5), str='%m-%d')
                setp(gca(), xticks=xts, xticklabels=xls, xlim=[stt, edt])
                if nn==0: 
                    text(xlim()[0] + 0.00 * diff(xlim()), ylim()[0] + 1.1 * diff(ylim()),'{}({})-{}, {}, R: {:.2f}, RMSE: {:.2f}, ME: {:.2f}'.format(bp.staname[i],station,regname,lobs, st.R, st.RMSD, MEi), color='k', fontweight='bold')
            legend(['Obs.',*tags])
            xts, xls = get_xtick(fmt=2, xts=linspace(stt,edt,5), str='%m-%d')
            setp(gca(), xticks=xts, xticklabels=xls, xlim=[stt, edt])
            gca().xaxis.grid('on')
            gca().yaxis.grid('on')
            gcf().tight_layout()

            savefig('{}/{}_{}.png'.format(sname,regname, bp.staname[i]), dpi=450, bbox_inches='tight')
            close()
                # figure(20)
                # plot(bp.x[i],bp.y[i],'r.')

            # if region=='SAB':
            #     setp(gca(), ylim=[-1.5,3], yticks=[-1.5, 0, 1.5, 3]) # SAB
            #     if iflag < maxfig - 1: setp(gca(), xticklabels=[]);
            #     if iflag > maxfig - 2: xlabel('Date (2016-10-)')

            # else:
            #     setp(gca(), ylim=[-1.2,1.2], yticks=[-1.2, -0.6, 0, 0.6, 1.2]) # MAB
            #     if iflag < maxfig - 2: setp(gca(), xticklabels=[])
            #     if iflag > maxfig - 3: xlabel('Date (2016-10-)')






        # gcf().tight_layout()
        # savefig('{}_{}'.format(sname, region) + '.png', dpi=900, bbox_inches='tight')
        # close()
        # show()

MAE=array(MAE); Corr=array(Corr); ME=array(ME)
fpn=~isnan(MAE)
print('MAE: ', MAE[fpn].mean(axis=0))
print('Corr: ', Corr[fpn].mean(axis=0))
print('ME: ', ME[fpn].mean(axis=0))

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14
rc('font', family='Helvetica')
rc('font', size=SMALL_SIZE)  # controls default text sizes
rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
