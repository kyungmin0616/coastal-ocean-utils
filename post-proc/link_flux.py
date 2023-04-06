from pylib import *

dir_run='../../run'
dir_save='./'

fnames=array([i for i in os.listdir(dir_run) if i.startswith('RUN')])

for fname in fnames:
    if os.path.exists("{}/{}_flux.out".format(dir_save,fname)): print('{} exist'.format(fname)); continue
    os.system("ln -s {}/{}/outputs/flux.out {}/{}_flux.out".format(dir_run,fname,dir_save,fname))
