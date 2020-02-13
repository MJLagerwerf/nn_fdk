#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=0,1,2,3

for i in {0..20}
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp_DNNs_recons.py -p -F \
    NNFDK_results/DNN_stats_4S with phantom='Fourshape' --log=WARNING
done

for i in {0..20}
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python exp_DNNs_recons.py -p -F \
    NNFDK_results/DNN_stats_DF with phantom='Defrise random' --log=WARNING
done

