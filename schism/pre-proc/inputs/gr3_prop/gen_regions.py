from pylib import *

# regions to be selected
gd=read('./hgrid.gr3')
bp=read('reg3.reg')
sindp=inside_polygon(c_[gd.x,gd.y],bp.x,bp.y)==1

gd.dp[:]=1
gd.dp[sindp]=3

gd.write('v_regions.gr3')
