#!/bin/bash
d=20190601
while [ "$d" != 20190611 ]; do
  echo $d
  python JUNO/src/CCA_frontal_prob_xarray.py 'sst_'"$d"'.nc'
  d=$(date +%Y%m%d -d "$d + 1 day")
done

