#!/bin/bash

TNAME='hycom_2016_09_08_00.nc'
cdo sellonlatbox,-110,0,0,70 $TNAME tmp.nc
cp tmp.nc $TNAME
rm tmp.nc
ncatted -a missing_value,,m,f,-30000 $TNAME
ncatted -a _FillValue,,m,f,-30000 $TNAME

SSH="SSH.nc"
TS="TS.nc"
UV="UV.nc"
#ncrename -v water_temp,temperature -v lon,xlon -v lat,xlat -d lon,xlon -d lat,xlat  $TNAME tmp.nc
ncrename -v water_temp,temperature $TNAME tmp.nc

ncks -v lon,lat,time,surf_el tmp.nc $SSH
ncks -v lon,lat,depth,time,temperature,salinity tmp.nc $TS
ncks -v lon,lat,depth,time,water_v,water_u tmp.nc $UV
rm tmp.nc

#unpack & cvt to float
#SSH file
ncpdq -O -U $SSH tmp1.nc
cdo -b f32 copy tmp1.nc tmp2.nc
ncks -O --mk_rec_dmn time tmp2.nc -o tmp3.nc
cdo chname,lon,xlon,lat,ylat tmp3.nc tmp4.nc
mv tmp4.nc SSH_1.nc
rm $SSH tmp1.nc tmp2.nc tmp3.nc

#ST file # this require cdo tool
ncpdq -O -U $TS tmp1.nc
cdo adipot tmp1.nc tmp2.nc
ncrename -v tho,temperature -v s,salinity tmp2.nc tmp3.nc
cdo -b f32 copy tmp3.nc tmp4.nc
ncks -O --mk_rec_dmn time tmp4.nc -o tmp5.nc
cdo chname,lon,xlon,lat,ylat tmp5.nc tmp6.nc
mv tmp6.nc TS_1.nc
rm $TS tmp1.nc tmp2.nc tmp3.nc tmp4.nc tmp5.nc

#UV file
ncpdq -O -U $UV tmp1.nc
cdo -b f32 copy tmp1.nc tmp2.nc
ncks -O --mk_rec_dmn time tmp2.nc -o tmp3.nc
cdo chname,lon,xlon,lat,ylat tmp3.nc tmp4.nc
mv tmp4.nc UV_1.nc
rm $UV tmp1.nc tmp2.nc tmp3.nc
