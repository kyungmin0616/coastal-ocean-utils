#!/usr/bin/env python3
from pylib import *

#input
bdir='/home1/08924/kmpark/UFS/'
grid='02'
exp='b'

## making folders 
if os.path.exists("{}/run/RUN{}{}".format(bdir,grid,exp)):
        os.system("rm -r {}/run/RUN{}{}; mkdir {}/run/RUN{}{}".format(bdir,grid,exp,bdir,grid,exp))
else: os.system("mkdir {}/run/RUN{}{}".format(bdir,grid,exp))

#if os.path.exists("{}/outputs/RUN{}{}".format(bdir,grid,exp)):
#       os.system("rm -r {}/outputs/RUN{}{}; mkdir {}/outputs/RUN{}{}".format(bdir,grid,exp,bdir,grid,exp))
#else:os.system("mkdir {}/outputs/RUN{}{}".format(bdir,grid,exp))

# outputs
os.system("mkdir {}/run/RUN{}{}/outputs".format(bdir,grid,exp))
os.system("touch {}/run/RUN{}{}/outputs/flux.out".format(bdir,grid,exp))

##in run

#grid
os.system("cp {}/pre-proc/grid/{}/hgrid.gr3 {}/run/RUN{}{}/.".format(bdir,grid,bdir,grid,exp))
os.system("cp {}/pre-proc/grid/{}/hgrid.ll {}/run/RUN{}{}/.".format(bdir,grid,bdir,grid,exp))
os.system("cp {}/pre-proc/grid/{}/vgrid.in {}/run/RUN{}{}/.".format(bdir,grid,bdir,grid,exp))
os.system("cp {}/pre-proc/grid/{}/hgrid.utm {}/run/RUN{}{}/.".format(bdir,grid,bdir,grid,exp))

#inputs
fnames=array([i for i in os.listdir('{}/pre-proc/inputs/RUN{}{}/gr3_prop/'.format(bdir,grid,exp)) if i.endswith(tuple(["prop","gr3"]))])
if len(fnames):
        for fname in fnames:
                os.system("cp {}/pre-proc/inputs/RUN{}{}/gr3_prop/{} {}/run/RUN{}{}/.".format(bdir,grid,exp,fname,bdir,grid,exp))

fnames=array([i for i in os.listdir('{}/pre-proc/inputs/RUN{}{}/bctides/'.format(bdir,grid,exp)) if i.endswith(tuple(["in"]))])
if len(fnames):
        for fname in fnames:
                os.system("cp {}/pre-proc/inputs/RUN{}{}/bctides/{} {}/run/RUN{}{}/.".format(bdir,grid,exp,fname,bdir,grid,exp))

