#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
#export CUDA_VISIBLE_DEVICES=2,3

for i in {0..3}
do
    python exp_real_data.py -p -F \
    NNFDK_results/real_data with it_i=$i nTD=10 nVD=5
done


python exp_real_data_CA.py -p -F NNFDK_results/real_data_CA with it_i=$i nTD=10 nVD=5

