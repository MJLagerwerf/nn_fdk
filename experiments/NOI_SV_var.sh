#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
#export CUDA_VISIBLE_DEVICES=2,3


for i in {0..9}
do
    python NOI_var.py -p -F \
    NNFDK_results/NOI_var_1024 with it_i=$i pix=1024
done

for i in {0..8}
do
    python SV_var.py -p -F \
    NNFDK_results/SV_var_1024 with it_i=$i pix=1024
done

for i in 1 2 3 5 7.5 10
do
    python exp_cone_angle.py -p with it_i=$i pix=1024 angle=360
done