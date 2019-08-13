#!/usr/bin/bash
# limit numper of OpenMP threads
# export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=0,1


for i in {5..8}
do
    python SV_var.py -p -F \
    NNFDK_results/SV_var_1024 with it_i=$i pix=1024
done

