from pylib import *

dir_data='../NC4'

fnames=array([i for i in os.listdir(dir_data) if i.endswith('.nc')])

for fname in fnames:
    os.system("nccopy -k 1 {}/{} ./{}".format(dir_data,fname,fname))

