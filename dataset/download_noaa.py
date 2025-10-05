#!/usr/bin/env python3
from pylib import *

#-----------------------------------------------------
#download NOAA elevation data:  NAVD and MSL
#-----------------------------------------------------
#input
StartT=datenum(2021,6,1); EndT=datenum(2021,11,1)
datums=['navd','msl']
station_list='stations.txt'

#noaa web link: if download not work, check this #water_level, predictions, currents, currents_predictions, wind, air_pressure
url0=r'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=water_level&application=NOS.COOPS.TAC.WL&time_zone=GMT&units=metric&format=csv'

#read all station info
C=npz_data(); C.lon,C.lat,C.station=x=loadtxt(station_list,skiprows=1).T
C.station=C.station.astype('int')

#download and read noaa data
for datum in datums:
    if not os.path.exists(datum): os.mkdir(datum)

    #download data
    for m,station in enumerate(C.station):
        y1=num2date(StartT).year; y2=num2date(EndT).year
        m1=num2date(StartT).month; m2=num2date(EndT).month

        for year in arange(y1,y2+1):
            m1i=m1 if year==y1 else 1
            m2i=m2 if year==y2 else 12
            for month in arange(m1i,m2i+1):
                StartTi=datenum(year,month,1)
                EndTi=datenum(year+1,1,1)-1 if month==12 else datenum(year,month+1,1)-1
                #download each month data
                url='{}&datum={}&begin_date={}&end_date={}&station={}'.format(url0,datum.upper(),num2date(StartTi).strftime('%Y%m%d'),num2date(EndTi).strftime('%Y%m%d'),station)
                fname='{}_{}_{:02}.csv'.format(station,year,month)

                if os.path.exists('{}/{}'.format(datum,fname)): continue
                print('download: {}, {}, {}'.format(datum,fname, m))
                try:
                    urllib.request.urlretrieve(url,'{}/{}'.format(datum,fname))
                except:
                    pass

    #read data
    fnames=os.listdir('{}'.format(datum));
    #read each file in years
    mtime=[]; station=[]; elev=[]; iflag=0
    for fname in fnames:
        if not fname.endswith('.csv'): continue
        R=re.match('(\d+)_(\d+)_(\d+).csv',fname); sta=int(R.groups()[0]); year=int(R.groups()[1]); month=int(R.groups()[2])

        #read data
        iflag=iflag+1; print('reading {}, {}'.format(fname,iflag))
        fid=open('{}/{}'.format(datum,fname),'r'); lines=fid.readlines(); fid.close(); lines=lines[1:]
        if len(lines)<10: continue

        #parse each line
        for i in arange(len(lines)):
            line=lines[i].split(',')
            if line[1]=='': continue
            doyi=datestr2num(line[0]); elevi=float(line[1])

            #save record
            mtime.append(doyi)
            station.append(sta)
            elev.append(elevi)

    #-save data-------
    S=npz_data(); S.time=array(mtime); S.elev=array(elev)
    S.station=array(station).astype('int')

    # add lat&lon information
    Lat=dict(zip(C.station,C.lat)); Lon=dict(zip(C.station,C.lon))
    S.lat=array([Lat[i] for i in S.station])
    S.lon=array([Lon[i] for i in S.station])
    save_npz('noaa_elev_{}'.format(datum),S)

print('-------done------')
