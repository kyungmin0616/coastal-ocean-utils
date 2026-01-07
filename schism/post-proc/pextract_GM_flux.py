#!/usr/bin/env python3
'''
Compute fluxes based on SCHISM node information
'''
from pylib import *
import time

#-----------------------------------------------------------------------------
#Input
#-----------------------------------------------------------------------------
run='GM-flux'
chpt=0; # 1: check points of transect

# GM info
# CMEMS
#sname='./CMEMS_flux_2013'
#dir_data='/rcfs/projects/mhk_modeling/dataset/CMEMS/NAO/reanalysis/'
#variables=['longitude','latitude','depth','uo','vo'] # CMEMS

#HYCOM
sname='./npz/HYCOM_flux_2013'
dir_data='/rcfs/projects/mhk_modeling/dataset/HYCOM/FC/'
variables=['lon','lat','depth','water_u','water_v'] # HYCOM

txy=[[[-521.402671, 120000], [2963981.428258, 2959984.100407]]] # UTM 18

StartT=datenum(2013,1,1); EndT=datenum(2014,1,1)

subdm=[-82.9,-74.9,20.3,33.6]

#optional
dx=3000          #interval of sub-section, used to divide transect
#prj='cpp'      #projection that convert lon&lat to local project when ics=2
#rvars=['g1','g2','g3',] #rname the varibles

#resource requst 
walltime='02:00:00'
qnode='deception'; nnode=1; ppn=8  #frontera, ppn=56 (flex,normal)

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
if any(S2.variables[variables[0]][:]>180): 
    sxp=(S2.variables[variables[0]][:]+180)%360-180; lonidx=argsort(sxp); sxp=sxp[lonidx]
else:
    sxp=S2.variables[variables[0]][:]
    lonidx=None
syp=S2.variables[variables[1]][:]
sloidx=(subdm[0]<sxp)*(sxp<subdm[1])
slaidx=(subdm[2]<syp)*(syp<subdm[3])
sxp=sxp[sloidx]; syp=syp[slaidx]
sx,sy=meshgrid(sxp,syp); sx=sx.ravel(); sy=sy.ravel(); sx,sy=proj_pts(sx,sy,'epsg:4326','epsg:26918')
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
    ang=array([arctan2(yi[i+1]-yi[i],xi[i+1]-xi[i]) for i in arange(npt-1)])   #angle for each subsection
    nps.append(npt); pxy[m]=c_[xi,yi].T; dsa.append(ds); #sx.extend(xi); sy.extend(yi)
    sinds.append(arange(ipt,ipt+npt)); angles.append(ang); ipt=ipt+npt

#check selected points
if chpt==1:
    eta=S2.variables['surf_el'][:]
    clf();tricontourf(sx,sy,eta[0,:,:].data.ravel(),cmap='jet',levels=linspace(-2,2,51));plot(xi,yi,'r.');show()
    sys.exit()
#compute flux
S=zdata(); S.time=[]; S.flux=[[] for i in txy]
for nn,fname in enumerate(fnames):
    if nn%nproc!=myrank: continue
    t00=time.time(); S2=ReadNC('{}/{}'.format(dir_data,fname),1); 
    for m,npt in enumerate(nps): #for each transect
        if lonidx is not None:
            u=S2.variables[variables[3]][:,:,:,lonidx]; v=S2.variables[variables[4]][:,:,:,lonidx]; #read profile
        else:
            u=S2.variables[variables[3]]; v=S2.variables[variables[4]]
        u=u[:,:,:,sloidx][:,:,slaidx]; u=u.transpose(2,3,0,1); u=u.reshape(len(sx),len(S2.variables['time'][:]),len(S2.variables[variables[2]][:])); u=u[sindp,:]
        v=v[:,:,:,sloidx][:,:,slaidx]; v=v.transpose(2,3,0,1); v=v.reshape(len(sx),len(S2.variables['time'][:]),len(S2.variables[variables[2]][:])); v=v[sindp,:]
        fpt=u.mask==1; u[fpt]=NaN; fpt=v.mask==1; v[fpt]=NaN;
        u=squeeze(u); v=squeeze(v)

        sind=sinds[m]; ang=angles[m][:,None]; ds=dsa[m][:,None]
        depth=S2.variables[variables[2]][:]; #ndepth=arange(depth.min(),depth.max(),2) #fpt=depth>=100; ndepth=array([*depth[~fpt].data, *arange(100,5000,50)])
#        rdepth2=rdepth[sindp]; 
        nx,nz=meshgrid(sx[sindp],depth)
        nx=nx.transpose(); nz=nz.transpose() 
        #for nnn in arange(len(sindp)):
        #    nz[nnn,sum(~fpt[nnn])-1]=rdepth2[nnn]
        dnz=diff(nz)
        dnz=(dnz[:-1]+dnz[1:])/2
        u=(u[:-1,:-1]+u[:-1,1:]+u[1:,:-1]+u[1:,1:])/4; v=(v[:-1,:-1]+v[:-1,1:]+v[1:,:-1]+v[1:,1:])/4
#        ds=np.ones(shape(u))*ds
        #contourf(nx,nz,squeeze(v),cmap='jet',levels=50);colorbar();ylim([-800,0]);
        #savefig('./check_vf2/{}.png'.format(fname))
        #close()
        #volume flux 
        flx=array(((sin(ang)*u+cos(ang)*v)*dnz*ds).sum()); 
        S.flux[m].append(flx)
    S.time.extend(S2.variables['time'][:]); C=None
    print('Done {} on rank {}: {:0.2f}'.format(fname,myrank,time.time()-t00)); sys.stdout.flush()
S.time,S.flux=array(S.time),array(S.flux).T

#gather flux for all ranks
data=comm.gather(S,root=0)
C=zdata(); C.nps=array(nps); C.xy=pxy; C.xy0=txy; C.time,C.flux=[],[]
if myrank==0:
   for i in data: C.time.extend(i.time); C.flux.extend(i.flux)
   it=argsort(C.time); C.time=array(C.time)[it]; C.flux=array(C.flux)[it].T.astype('float32')

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
