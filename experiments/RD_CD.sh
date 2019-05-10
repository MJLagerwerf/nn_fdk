#!/usr/bin/bash
# limit numper of OpenMP threads
#export OMP_NUM_THREADS=16
# set astra gpu index: 0-3
#export CUDA_VISIBLE_DEVICES=2,3

for i in {1..21}
do
    python RD_CD.py -p -F \
    NNFDK_results/RD_CD with it_i=$i
done

