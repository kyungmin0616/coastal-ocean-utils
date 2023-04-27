from pylib import *
import time

#-----------------------------------------------------------------------------
#Input
#-----------------------------------------------------------------------------
run='CMEMS_elev'
StartT=datenum(2020,1,1); EndT=datenum(2020,3,4)
bpfile='./station_ver2.in'  #station file
sname='./HYCOM-elev-wo-tide-xyz'
subdm=[-105,-65,8,55]

############ GM info
# CMEMS
#dir_data='/rcfs/projects/mhk_modeling/dataset/CMEMS/NAO/reanalysis'
#variables=['longitude','latitude','zos'] # CMEMS
#HYCOM
dir_data='/rcfs/projects/mhk_modeling/dataset/HYCOM/NAO'
variables=['lon','lat','surf_el'] # HYCOM

#resource requst 
walltime='02:00:00'
#qnode='x5672'; nnode=2; ppn=8       #hurricane, ppn=8
#qnode='bora'; nnode=2; ppn=20      #bora, ppn=20
#qnode='vortex'; nnode=2; ppn=12    #vortex, ppn=12
#qnode='femto'; nnode=1; ppn=32     #femto,ppn=32
#qnode='potomac'; nnode=4; ppn=8    #ches, ppn=12
#qnode='james'; nnode=5; ppn=20     #james, ppn=20
qnode='frontera'; nnode=1; ppn=30  #frontera, ppn=56 (flex,normal)
#qnode='levante'; nnode=1; ppn=36   #levante, ppn=128
#qnode='stampede2'; nnode=1; ppn=48 #stampede2, ppn=48 (skx-normal,skx-dev,normal,etc)

#additional information:  frontera,levante,stampede2
qname='short'                        #partition name
account='MHK_MODELING'              #stampede2: NOAA_CSDL_NWI,TG-OCE140024; levante: gg0028 

brun=os.path.basename(run); jname='Rd_'+brun #job name
ibatch=0; scrout='screen_{}.out'.format(brun); bdir=os.path.abspath(os.path.curdir)
#-----------------------------------------------------------------------------
#on front node: 1). submit jobs first (qsub), 2) running parallel jobs (mpirun) 
#-----------------------------------------------------------------------------
if ibatch==0: os.environ['job_on_node']='1'; os.environ['bdir']=bdir #run locally
if os.getenv('job_on_node')==None:
   if os.getenv('param')==None: fmt=0; bcode=sys.argv[0]
   if os.getenv('param')!=None: fmt=1; bdir,bcode=os.getenv('param').split(); os.chdir(bdir)
   scode=get_hpc_command(bcode,bdir,jname,qnode,nnode,ppn,walltime,scrout,fmt=fmt,qname=qname)
   print(scode); os.system(scode); os._exit(0)

#-----------------------------------------------------------------------------
#on computation node
#-----------------------------------------------------------------------------
bdir=os.getenv('bdir'); os.chdir(bdir) #enter working dir
odir=os.path.dirname(os.path.abspath(sname))
comm=MPI.COMM_WORLD; nproc=comm.Get_size(); myrank=comm.Get_rank()
if myrank==0: t0=time.time()
if myrank==0 and (not fexist(odir)): os.mkdir(odir)

#-----------------------------------------------------------------------------
#do MPI work on each core
#-----------------------------------------------------------------------------

fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])
mti=array([datenum(*array(i.replace('.','_').split('_')[1:5]).astype('int')) for i in fnames])
fpt=(mti>=(StartT-1))*(mti<(EndT+1)); fnames=fnames[fpt]; mti=mti[fpt]
sind=argsort(mti); mti=mti[sind]; fnames=fnames[sind]

#check format of transects
bp=read_schism_bpfile(bpfile) 

# grid info for data
S2=ReadNC('{}/{}'.format(dir_data,fnames[0]),1)
if any(S2.variables[variables[0]][:]>180):
    sxp=(S2.variables[variables[0]][:]+180)%360-180; lonidx=argsort(sxp); sxp=sxp[lonidx]
else:
    sxp=(S2.variables[variables[0]][:])
    lonidx=None
syp=S2.variables[variables[1]][:]
sloidx=(subdm[0]<sxp)*(sxp<subdm[1])
slaidx=(subdm[2]<syp)*(syp<subdm[3])
sxp=sxp[sloidx]; syp=syp[slaidx]
sx,sy=meshgrid(sxp,syp); sx=sx.ravel(); sy=sy.ravel(); #sx,sy=proj_pts(sx,sy,'epsg:4326','epsg:26918')

#compute transect information
sindp=near_pts(c_[bp.x,bp.y],c_[sx,sy]); #sindp=unique(sindp)

#check selected points
#clf();tricontourf(sx,sy,S2.variables['zos'][:].data[:,:,sloidx][:,slaidx].ravel(),cmap='jet',levels=linspace(-2,2,51));plot(sx[sindp],sy[sindp],'r.');show()

#extract elev
S=zdata(); S.time=[]; S.elev=[]
for nn,fname in enumerate(fnames):
    if nn%nproc!=myrank: continue
    t00=time.time(); S2=ReadNC('{}/{}'.format(dir_data,fname),1);
    if lonidx is not None:
        elev=S2.variables[variables[2]][:,:,lonidx]; #u=S2.variables['uo'][:,:,:,lonidx]; v=S2.variables['vo'][:,:,:,lonidx]; #read profile
    else:
        elev=S2.variables[variables[2]]; #u=S2.variables['uo']; v=S2.variables['vo']
    elev=elev[0,:,sloidx][slaidx]; #u=u[:,:,:,sloidx][:,:,slaidx][:,0,:,:]; v=v[:,:,:,sloidx][:,:,slaidx][:,0,:,:] 
    elev=elev.ravel(); #u=u.ravel();v=v.ravel()
    fpt=elev.mask==1; elev=elev.data; elev[fpt]=NaN
    
    S.time.extend(S2.variables['time'][:]); S.elev.append(elev[sindp]);
    print('Done {} on rank {}: {:0.2f}'.format(fname,myrank,time.time()-t00)); sys.stdout.flush()
S.time,S.elev=array(S.time),array(S.elev)

#gather flux for all ranks
data=comm.gather(S,root=0)
C=zdata(); C.time,C.elev=[],[]
if myrank==0:
   for i in data: C.time.extend(i.time); C.elev.extend(i.elev)
   it=argsort(C.time); C.time=array(C.time)[it]; C.elev=array(C.elev)[it].T.astype('float32')

   #save
   sdir=os.path.dirname(os.path.abspath(sname))
   if not fexist(sdir): os.system('mkdir -p '+sdir)
   savez(sname,C)

#-----------------------------------------------------------------------------
#finish MPI jobs
#-----------------------------------------------------------------------------
comm.Barrier()
if myrank==0: dt=time.time()-t0; print('total time used: {} s'.format(dt)); sys.stdout.flush()
#sys.exit(0) if qnode in ['bora','levante'] else os._exit(0)

