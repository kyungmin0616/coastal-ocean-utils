#!/usr/bin/env python3
'''
  extract SCHISM slab outputs
'''
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pylib import *
import time

#-----------------------------------------------------------------------------
#Input
#-----------------------------------------------------------------------------
run='../../run/RUN12p'
sname='./npz/RUN12p-1lv-hvel-temp.npz'
svars=('hvel','temp')                 #variables to be extracted
levels=[1,]        #schism level indices (1-nvrt: surface-bottom; (>nvrt): kbp level)

#optional
stacks=[1,120]   #outputs stacks to be extracted
#nspool=12      #sub-sampling frequency within each stack (1 means all)
#mdt=1          #time window (day) for averaging output
#rvars=['elev','hvel','G1'] #rname the varibles 

#resource requst 
walltime='00:30:00'
qnode='frontera'; nnode=1; ppn=56  #frontera, ppn=56 (flex,normal)

#additional information:  frontera for MPI
qname='development'                   #partition name
account='OCE22003'              #account name
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
sdir=run+'/outputs'                            #output directory
if 'nspool' not in locals(): nspool=1          #subsample
if 'rvars' not in locals(): rvars=svars        #rename variables
if 'mdt' not in locals(): mdt=None             #rename variables
modules, outfmt, dstacks, dvars, dvars_2d = get_schism_output_info(sdir,1)  #schism outputs information
stacks=arange(stacks[0],stacks[1]+1) if ('stacks' in locals()) else dstacks #check stacks

#extract results
irec=0; oname=odir+'/.schout'
for svar in svars: 
   ovars=get_schism_var_info(svar,modules,fmt=outfmt)
   if ovars[0][1] not in dvars: continue 
   for istack in stacks:
       fname='{}_{}_{}_slab'.format(oname,svar,istack); irec=irec+1; t00=time.time()
       if irec%nproc==myrank: 
          read_schism_slab(run,svar,levels,istack,nspool,mdt,fname=fname)
          dt=time.time()-t00; print('finishing reading {}_{}.nc on myrank={}: {:.2f}s'.format(svar,istack,myrank,dt)); sys.stdout.flush()

#combine results
comm.Barrier()
if myrank==0:
   S=zdata(); S.time=[]; fnames=[]
   for i,[k,m] in enumerate(zip(svars,rvars)):
       data=[]; mtime=[]
       for istack in stacks:
           fname='{}_{}_{}_slab.npz'.format(oname,k,istack)
           if not fexist(fname): continue
           C=loadz(fname); data.extend(C.__dict__[k]); mtime.extend(C.time); fnames.append(fname)
       if len(data)>0: S.__dict__[m]=array(data)
       if len(mtime)>len(S.time): S.time=array(mtime)
   savez(sname,S)
   for i in fnames: os.remove(i)

#-----------------------------------------------------------------------------
#finish MPI jobs
#-----------------------------------------------------------------------------
comm.Barrier()
if myrank==0: dt=time.time()-t0; print('total time used: {} s'.format(dt)); sys.stdout.flush()
sys.exit(0) if qnode in ['bora','levante'] else os._exit(0)
