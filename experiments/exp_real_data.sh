#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
#export CUDA_VISIBLE_DEVICES=2,3

#for i in {0..3}
#do
#    CUDA_VISIBLE_DEVICES=2 \
#    python exp_real_data.py -p -F \
#    NNFDK_results/real_data_S3 with it_i=$i nTD=10 nVD=5
#done

CUDA_VISIBLE_DEVICES=2 python exp_RD_CA.py -p -F NNFDK_results/real_data_CA_S1 with it_i=$i nTD=1 nVD=0
CUDA_VISIBLE_DEVICES=2 python exp_RD_CA.py -p -F NNFDK_results/real_data_CA_S2 with it_i=$i nTD=1 nVD=1
CUDA_VISIBLE_DEVICES=2 python exp_RD_CA.py -p -F NNFDK_results/real_data_CA_S3 with it_i=$i nTD=10 nVD=5

