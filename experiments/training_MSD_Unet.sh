#!/bin/bash
echo "Give GPU index"
read GPU_i
# Experiment that I want to run
CUDA_VISIBLE_DEVICES=${GPU_i} python exp_Unet.py -p with exp_type='noise' it_i=5 

echo "Stopped training of exp type: noise, with it_i: 5, Unet"

# Experiment that I want to run
CUDA_VISIBLE_DEVICES=${GPU_i} python exp_MSD.py -p with exp_type=noise it_i=5 --log=WARNING

echo "Stopped training of exp type: noise, with it_i: 5, MSD"


