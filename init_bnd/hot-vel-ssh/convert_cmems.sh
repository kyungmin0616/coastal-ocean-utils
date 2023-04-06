#!/bin/bash

TNAME='cmems_2016_09_08_12.nc'
cdo sellonlatbox,-105,0,0,70 $TNAME tmp.nc
cp tmp.nc $TNAME
rm tmp.nc
ncap2 -O -s "depth(0)=0" $TNAME $TNAME
ncatted -a missing_value,,m,f,-30000 $TNAME
ncatted -a _FillValue,,m,f,-30000 $TNAME
#cdo -O invertlev $TNAME $TNAME

SSH="SSH.nc"
TS="TS.nc"
UV="UV.nc"
#ncrename -v water_temp,temperature -v lon,xlon -v lat,xlat -d lon,xlon -d lat,xlat  $TNAME tmp.nc
ncrename -v thetao,temperature -v zos,surf_el -v so,salinity -v uo,water_u -v vo,water_v $TNAME tmp.nc

ncks -v longitude,latitude,time,surf_el tmp.nc $SSH
ncks -v longitude,latitude,depth,time,temperature,salinity tmp.nc $TS
ncks -v longitude,latitude,depth,time,water_v,water_u tmp.nc $UV
rm tmp.nc

#unpack & cvt to float
#SSH file
ncpdq -O -U $SSH tmp1.nc
cdo -b f32 copy tmp1.nc tmp2.nc
ncks -O --mk_rec_dmn time tmp2.nc -o tmp3.nc
cdo chname,longitude,xlon,latitude,ylat tmp3.nc tmp4.nc
mv tmp4.nc SSH_1.nc
rm $SSH tmp1.nc tmp2.nc tmp3.nc

#ST file # this require cdo tool
ncpdq -O -U $TS tmp1.nc
cdo -b f32 copy tmp1.nc tmp2.nc
ncks -O --mk_rec_dmn time tmp2.nc -o tmp3.nc
cdo chname,longitude,xlon,latitude,ylat tmp3.nc tmp4.nc
mv tmp4.nc TS_1.nc
rm $TS tmp1.nc tmp2.nc tmp3.nc

#UV file
ncpdq -O -U $UV tmp1.nc
cdo -b f32 copy tmp1.nc tmp2.nc
ncks -O --mk_rec_dmn time tmp2.nc -o tmp3.nc
cdo chname,longitude,xlon,latitude,ylat tmp3.nc tmp4.nc
mv tmp4.nc UV_1.nc
rm $UV tmp1.nc tmp2.nc tmp3.nc
