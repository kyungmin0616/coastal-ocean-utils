from pylib import *

#####################
grd='../../../grid/01/hgrid.gr3'
### *.ic
icf=0 # 1: make *.ic, 0: don't

### *.gr3
#manning coef
m_depths=[-1,-3]       #two hgrid depth(m) to distinguish river, land and the transition zone
mvalues=[0.025,0.04]  #lower and upper limits of manning values
m_regions=None
m_rvalues=(0.2,0.2,0.005,0.005,0.005)

#shapiro coef
shapiro_max=0.4
threshold_slope=0.75
s_depths=[-99999, 20]  # tweaks in shallow waters
shapiro_vals1=[0.2, 0.2]  # tweaks in shallow waters
s_regions=None  # tweaks in regions, the order matters
shapiro_vals2=None  # tweaks in regions, the order matters
i_set_add_s=None  # 0: set; 1: add

## Property
#fluxflag
f_regions=None 

#####################
gd=read_schism_hgrid(grd)
dp=gd.dp.copy()

# *.gr3
gd.write_hgrid('./estuary.gr3',value=0)
gd.write_hgrid('./drag.gr3',value=0.0025)
gd.write_hgrid('./diffmin.gr3',value=1e-6)
gd.write_hgrid('./diffmax.gr3',value=1)
gd.write_hgrid('./rough.gr3',value=0.001)
gd.write_hgrid('./albedo.gr3',value=1e-1)
gd.write_hgrid('./watertype.gr3',value=1) 
gd.write_hgrid('./windrot_geo2proj.gr3',value=0)
if icf==1:
    gd.write_hgrid('./salt.ic',value=0)
    gd.write_hgrid('./temp.ic',value=20)
    gd.write_hgrid('./elev.ic',value=0)

# Manning coef
if m_depths is not None:
    #compute manning coefficients
    mval=mvalues[0]+(dp-m_depths[0])*(mvalues[1]-mvalues[0])/(m_depths[1]-m_depths[0])
    fpm=mval<mvalues[0]; mval[fpm]=mvalues[0]
    fpm=mval>mvalues[1]; mval[fpm]=mvalues[1]

    #set values in regions
    if m_regions is not None:
        for m,m_region in enumerate(m_regions):
            print('modifying manning in {}'.format(m_region))
            bp=read_schism_bpfile(m_region,fmt=1)
            sind=inside_polygon(c_[gd.x,gd.y], bp.x,bp.y)
            fp=sind==1; mval[fp]=m_rvalues[m]

    #save manning.gr3
    gd.dp=mval
    gd.write_hgrid('manning.gr3')

gd.dp=dp
# Shapiro coef
if shapiro_max is not None:
    # compute bathymetry gradient on each node
    _, _, slope = gd.compute_gradient(fmt=2,cpp=1)

    # compute shapiro coefficients
    shapiro=shapiro_max*tanh(2*slope/threshold_slope)

    # further tweaks on shallow waters
    if s_depths is not None:
        if len(s_depths) != len(shapiro_vals1):
           raise Exception(f'lengths of depths {len(depths)} and shapiro_vals1 {len(shapiro_vals1)} inconsistent')
        fp = dp < s_depths[-1]
        shapiro[fp] = maximum(shapiro[fp], interp(dp[fp], s_depths, shapiro_vals1))

    #set or add values in regions
    if s_regions is not None:
        for i_set_add, s_rvalue, s_region in zip(i_set_add_s, shapiro_vals2, s_regions):
            bp=read_schism_bpfile(s_region,fmt=1)
            sind=inside_polygon(c_[gd.x,gd.y], bp.x,bp.y).astype('bool')

            if i_set_add==0:
                print(f'setting {s_rvalue} shapiro in {s_region}')
                fp=sind
                shapiro[fp]=s_rvalue
            else:
                print(f'adding {s_rvalue} shapiro in {s_region}')
                sind2=(dp>s_depths[0])  # additional condition: deeper water, dp > -1 m
                fp=(sind & sind2)
                shapiro[fp]=shapiro[fp]+rvalue

    #save shapiro.gr3
    gd.dp=shapiro
    gd.write_hgrid('shapiro.gr3')

##### *.prop

#Fluxflag
if f_regions is not None:
    gd=read_schism_hgrid(grd)
    pvi=-ones(gd.ne).astype('int'); gd.compute_ctr()
    for m,f_region in enumerate(f_regions):
        #read region info
        bp=read_schism_bpfile(f_region,fmt=1)
        if bp.nsta!=4: sys.exit(f'{region}''s npt!=4')
        x1,x2,x3,x4=bp.x; y1,y2,y3,y4=bp.y

        #middle pts
        mx1=(x1+x4)/2; mx2=(x2+x3)/2
        my1=(y1+y4)/2; my2=(y2+y3)/2

        #for lower region
        px=array([mx1,mx2,x3,x4]); py=array([my1,my2,y3,y4])
        pvi[inside_polygon(c_[gd.xctr,gd.yctr],px,py)==1]=m

        #for upper region
        px=array([mx1,x1,x2,mx2]); py=array([my1,y1,y2,my2])
        pvi[inside_polygon(c_[gd.xctr,gd.yctr],px,py)==1]=m+1
    gd.write_prop('fluxflag.prop',value=pvi,fmt='{:3d}')

# tvd.prop
gd.write_prop('./tvd.prop',value=1)
print('---------done--------')
