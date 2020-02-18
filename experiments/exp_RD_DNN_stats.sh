#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=0,1,2,3

for i in {16..20}
do
    python exp_RD_DNNs_recon.py -p -F \
    NNFDK_results/DNN_stats_noisy with it_i=0 it_j=$i --log=WARNING
done

for i in {16..20}
do
    python exp_RD_DNNs_recon.py -p -F \
    NNFDK_results/DNN_stats_good with  it_i=1 it_j=$i --log=WARNING
done

