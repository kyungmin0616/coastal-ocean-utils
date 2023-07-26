#!/usr/bin/env python3
'''
Compute fluxes based on SCHISM node information
'''
from pylib import *
import time

#-----------------------------------------------------------------------------
#Input
#-----------------------------------------------------------------------------
run='GM-vert'
chpt=0; # 1: check points of transect

# GM info
# CMEMS
#sname='./CMEMS_flux_2013'
#dir_data='/rcfs/projects/mhk_modeling/dataset/CMEMS/NAO/reanalysis/'
#variables=['longitude','latitude','depth','uo','vo'] # CMEMS

#HYCOM
sname='HYCOM_vet_TS_FC_2013'
dir_data='/rcfs/projects/mhk_modeling/dataset/HYCOM/FC/'
rvars=['water_temp','salinity','water_u','water_v'] # HYCOM
svars=['temp','salt','u','v']

coord=['lon','lat','depth']
tcvt=24
reftime=datenum(2000,1,1)

#txy=[[[-521.402671, 120000], [2963981.428258, 2959984.100407]]] # UTM 18
txy='transect.bp'
StartT=datenum(2013,1,1); EndT=datenum(2013,2,1)

subdm=[-82.9,-74.9,20.3,33.6]

#optional
#dx=3000          #interval of sub-section, used to divide transect
#prj='cpp'      #projection that convert lon&lat to local project when ics=2
#rvars=['g1','g2','g3',] #rname the varibles

#resource requst 
walltime='02:00:00'
qnode='deception'; nnode=1; ppn=16  #frontera, ppn=56 (flex,normal)

#additional information:  frontera,levante,stampede2
qname='slurm'    #partition name
account='MHK_MODELING'   #stampede2: NOAA_CSDL_NWI,TG-OCE140024; levante: gg0028

brun=os.path.basename(run); jname='Rd_'+brun #job name 
ibatch=1; scrout='screen_{}.out'.format(run); bdir=os.path.abspath(os.path.curdir)
#-----------------------------------------------------------------------------
#on front node: 1). submit jobs first (qsub), 2) running parallel jobs (mpirun) 
#-----------------------------------------------------------------------------
if ibatch==0: os.environ['job_on_node']='1'; os.environ['bdir']=bdir #run locally
if os.getenv('job_on_node')==None:
   if os.getenv('param')==None: fmt=0; bcode=sys.argv[0]
   if os.getenv('param')!=None: fmt=1; bdir,bcode=os.getenv('param').split(); os.chdir(bdir)
   scode=get_hpc_command(bcode,bdir,jname,qnode,nnode,ppn,walltime,scrout,fmt=fmt,qname=qname,account=account)
   print(scode); os.system(scode); os._exit(0)

#-----------------------------------------------------------------------------
#on computation node
#-----------------------------------------------------------------------------
bdir=os.getenv('bdir'); os.chdir(bdir) #enter working dir
comm=MPI.COMM_WORLD; nproc=comm.Get_size(); myrank=comm.Get_rank()
if myrank==0: t0=time.time()

#-----------------------------------------------------------------------------
#do MPI work on each core
#-----------------------------------------------------------------------------
fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])
mti=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

#check format of transects
if isinstance(txy,str) or array(txy[0]).ndim==1: txy=[txy]
rdp=read_schism_bpfile; txy=[[rdp(i).x,rdp(i).y] if isinstance(i,str) else i for i in txy]

# grid info for data
S2=ReadNC('{}/{}'.format(dir_data,fnames[0]),1)
if any(S2.variables[coord[0]][:]>180): 
    sxp=(S2.variables[coord[0]][:]+180)%360-180; lonidx=argsort(sxp); sxp=sxp[lonidx]
else:
    sxp=S2.variables[coord[0]][:]
    lonidx=None
syp=S2.variables[coord[1]][:]
sloidx=(subdm[0]<sxp)*(sxp<subdm[1])
slaidx=(subdm[2]<syp)*(syp<subdm[3])
sxp=sxp[sloidx]; syp=syp[slaidx]
sx,sy=meshgrid(sxp,syp); sx=sx.ravel(); sy=sy.ravel(); #sx,sy=proj_pts(sx,sy,'epsg:4326','epsg:26918')
#S3=ReadNC('depth_GOMu0.04_03i.nc');rdepth=S3.depth.val.ravel()

#compute transect information
nps,dsa=[],[]; sinds,angles=[],[]; ns=len(txy); pxy=ones(ns).astype('O'); ipt=0
for m,[x0,y0] in enumerate(txy):
    #compute transect pts
    x0=array(x0); y0=array(y0)
    if 'dx' in locals():  #divide transect evenly
       ds=abs(diff(x0+1j*y0)); s=cumsum([0,*ds]); npt=int(s[-1]/dx)+1; ms=linspace(0,s[-1],npt)
       xi=interpolate.interp1d(s,x0)(ms); yi=interpolate.interp1d(s,y0)(ms)
    else:
       xi,yi=x0,y0;
    sindp=near_pts(c_[xi,yi],c_[sx,sy]); sindp=unique(sindp)
    if sum(sindp==0)!=0: sys.exit('transect pts outside of domain')
    if 'prj' in locals(): pxi,pyi=proj_pts(xi,yi,'epsg:4326',prj); ds=abs(diff(pxi+1j*pyi))
    xi=sx[sindp]; yi=sy[sindp]; npt=len(xi); ds=abs(diff(xi+1j*yi));
    #transect property
    #ang=array([arctan2(yi[i+1]-yi[i],xi[i+1]-xi[i]) for i in arange(npt-1)])   #angle for each subsection
    nps.append(npt); pxy[m]=c_[xi,yi].T; #dsa.append(ds); #sx.extend(xi); sy.extend(yi)
    sinds.append(arange(ipt,ipt+npt)); #angles.append(ang); ipt=ipt+npt

#check selected points
if chpt==1:
    eta=S2.variables['surf_el'][:]
    clf();tricontourf(sx,sy,eta[0,:,:].data.ravel(),cmap='jet',levels=linspace(-2,2,51));plot(xi,yi,'r.');show()
    sys.exit()
#compute flux
S=zdata(); S.time=[]; [exec('S.{}=[[] for i in txy]'.format(i)) for i in svars]
for nn,fname in enumerate(fnames):
    #if nn%nproc!=myrank: continue
    print(fname); t00=time.time(); S2=ReadNC('{}/{}'.format(dir_data,fname),1); 
    for m,npt in enumerate(nps): #for each transect
        for nn,rvar in enumerate(rvars):
        
            if lonidx is not None:
                tmp=S2.variables[rvar][:,:,:,lonidx]; #read profile
            else:
                tmp=S2.variables[rvar]; 
            tmp=tmp[:,:,:,sloidx][:,:,slaidx]; tmp=tmp.transpose(2,3,0,1); tmp=tmp.reshape(len(sx),len(S2.variables['time'][:]),len(S2.variables[coord[2]][:])); tmp=tmp[sindp,:]
            fpt=tmp.mask==1; tmp[fpt]=NaN; #masking data
            tmp=squeeze(tmp) 
            exec('S.{}[m].append(tmp)'.format(svars[nn]))
    S.time.extend(S2.variables['time'][:]/tcvt+reftime); C=None
    print('Done {} on rank {}: {:0.2f}'.format(fname,myrank,time.time()-t00)); sys.stdout.flush()

for m,npt in enumerate(nps):
    for nn,rvar in enumerate(rvars):
        exec('S.{}[m]=array(S.{}[m])'.format(svars[nn],svars[nn]))

#gather flux for all ranks
data=comm.gather(S,root=0)
C=zdata(); C.nps=array(nps); C.lon=array(xi); C.lat=array(xi); C.xy0=txy; C.time=[]; [exec('C.{}=empty((ns,),dtype=object)'.format(i)) for i in rvars];
S.depth=array(S2.variables[coord[2]][:])
if myrank==0:
   for i in data:
       C.time.extend(i.time); #C.flux.extend(i.flux); tflux.extend(i.tflux)
       for svar in svars:
           for m,npt in enumerate(nps):
               exec('C.{}[m].extend(i.{}[m])'.format(svar,svar))
   it=argsort(C.time); C.time=array(C.time)[it]; #C.flux=array(C.flux)[it].T.astype('float32') 
   for svar in svars:
    for m,npt in enumerate(nps):
        exec('C.{}[m]=array(C.{}[m])[it]'.format(svar,svar))

   #save
   sdir=os.path.dirname(os.path.abspath(sname))
   if not fexist(sdir): os.system('mkdir -p '+sdir)
   savez(sname,C)


#-----------------------------------------------------------------------------
#finish MPI jobs
#-----------------------------------------------------------------------------
comm.Barrier()
if myrank==0: dt=time.time()-t0; print('total time used: {} s'.format(dt)); sys.stdout.flush()
#sys.exit(0) if qnode in ['bora'] else os._exit(0)
