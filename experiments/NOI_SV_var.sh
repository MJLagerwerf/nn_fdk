#!/usr/bin/bash
# limit numper of OpenMP threads
# export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
# export CUDA_VISIBLE_DEVICES=3


#for i in {0..4}
#do
#    CUDA_VISIBLE_DEVICES=3 \
#    python SV_var.py -p -F \
#    /export/scratch2/lagerwer/NNFDK_results/SV_var_1024 with it_i=$i pix=1024 nTD=10 nVD=5
#done

for i in {0..5}
do
    CUDA_VISIBLE_DEVICES=3 \
    python NOI_var.py -p -F \
    /export/scratch2/lagerwer/NNFDK_results/NOI_var_1024_S1 with it_i=$i pix=1024 nTD=1 nVD=0
done

for i in {0..5}
do
    CUDA_VISIBLE_DEVICES=3 \
    python NOI_var.py -p -F \
    /export/scratch2/lagerwer/NNFDK_results/NOI_var_1024_S2 with it_i=$i pix=1024 nTD=1 nVD=1
done
#for i in 2 3 5 7.5 10
#do
#    CUDA_VISIBLE_DEVICES=3 \
#    python exp_cone_angle.py -p with src_rad=$i pix=1024 nTD=10 nVD=5
#done
