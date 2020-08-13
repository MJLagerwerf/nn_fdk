#!/bin/bash
echo "Do you want a timer with this one? [y, n]"
read timer_bool

echo "Give GPU index"
read GPU_i

#echo "Specify which exp types you want, 0 - angles, 1 - cone angle, 2 - noise"
#read -a inds

echo "Experiment iteration"
read ExpI

ExpT=("'cone angle'" "'noise'")

if [ $timer_bool == 'y' ]; then 
	# Experiment that I want to run
	CUDA_VISIBLE_DEVICES=${GPU_i} python exp_Unet.py -p with exp_type='cone angle' it_i=${ExpI} &
	# Sleeps how much time you need:
	sleep 2d

	# Kills all python processes:
	nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

	echo "Stopped training of exp type: cone angle, with it_i: ${ExpI}, Unet"

	# Experiment that I want to run
	CUDA_VISIBLE_DEVICES=${GPU_i} python exp_Unet.py -p with exp_type='noise' it_i=${ExpI}  &
	# Sleeps how much time you need:
	sleep 2d

	# Kills all python processes:
	nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

	echo "Stopped training of exp type: noise, with it_i: ${ExpI}, Unet"
else
	# Experiment that I want to run
	CUDA_VISIBLE_DEVICES=${GPU_i} python exp_Unet.py -p with exp_type='cone angle' it_i=${ExpI}

	echo "Stopped training of exp type: cone angle, with it_i: ${ExpI}, Unet"

	CUDA_VISIBLE_DEVICES=${GPU_i} python exp_Unet.py -p with exp_type='noise' it_i=${ExpI} 
	echo "Stopped training of exp type: noise, with it_i: ${ExpI}, Unet"
fi

