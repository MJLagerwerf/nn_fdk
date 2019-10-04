#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
#export CUDA_VISIBLE_DEVICES=2,3

for i in {01..21}
do
    python exp_make_FDKs_RD.py -p -F  \
    /export/scratch2/lagerwer/NNFDK_results/FDK_RD/ with ang_freq=4 it_i=$i
done


