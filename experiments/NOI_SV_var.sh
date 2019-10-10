#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
#export CUDA_VISIBLE_DEVICES=2,3


for i in {1 2 3 5 7.5 10}
do
    python exp_cone_angle.py -p with src_rad=$i pix=1024
done

