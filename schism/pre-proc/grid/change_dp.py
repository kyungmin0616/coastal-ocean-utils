import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pylib import *

grd='hgrid.ll.new'
fname='hgrid.ll'

#regions=['min_h_harlem.reg','min_h_BronxKill.reg','min_h_Hudson.reg','min_h_Passaic_Hackensack.reg','min_h_Rah.reg','min_h_rari.reg','min_h_south_bound.reg','min_h_north_bound.reg','tri1.reg','tri2_permont.reg','harlem_battery_eastriver.reg','kingspoint.reg']  # tweaks in regions, the order matters
regions=['min_h_harlem.reg','min_h_BronxKill.reg','min_h_Hudson.reg','min_h_Passaic_Hackensack.reg','min_h_Rah.reg','min_h_rari.reg','min_h_south_bound.reg','min_h_north_bound.reg','tri1.reg','tri2_permont.reg','min_h_north_bound2.reg','min_h_north_bound3.reg','min_h_north_bound4.reg','min_h_north_bound5.reg','min_h_north_bound6.reg']
vals=[0.5,5,5,5,5,5,20,55,5,5,38,62,50,58,58]  # tweaks in regions, the order matters
i_set_add_s=[0,1,1,1,1,1,0,0,0,0,0,0,0,0,58]  # 0: gd.dp<=0; 1: gd.dp <=vals


#read hgrid
if grd.endswith('.npz'):
    gd=loadz(grd).hgrid
else:
    gd=read_schism_hgrid(grd)

#set or add values in regions
if regions is not None:
    for i_set_add, rvalue, region in zip(i_set_add_s, vals, regions):
        print(region)
        bp=read_schism_bpfile(region,fmt=1)
        sind=inside_polygon(c_[gd.x,gd.y], bp.x,bp.y).astype('bool')
        # check
#        clf()
#        gd.plot()
#        plot(gd.x[sind],gd.y[sind],'r.')
#        show()

        dp=gd.dp[sind].copy()
        if i_set_add==0:
            print(f'Dry correction: setting {rvalue} depth in {region}')
            fpt=dp<=0
            dp[fpt]=rvalue
            gd.dp[sind]=dp
        else:
            print(f'Min depth: setting {rvalue} depth in {region}')
            fpt=dp<rvalue
            dp[fpt]=rvalue
            gd.dp[sind]=dp
gd.write_hgrid(fname=fname)
