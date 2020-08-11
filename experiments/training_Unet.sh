#!/bin/bash
echo "Do you want a timer with this one? [y, n]"
read timer_bool

echo "Give GPU index"
read GPU_i

echo "Give experiment types in array form with ' and spaces in between"
read -a ExpT

echo "Experiment iteration"
read ExpI

for i in ${ind[@]}; 
do
	if [ $timer_bool == 'y' ]; then 
		# Experiment that I want to run
		CUDA_VISIBLE_DEVICES=${GPU_i} python exp_Unet.py -p with exp_type=${ExpT[$i} it_i=${ExpI} &
		# Sleeps how much time you need:
		sleep 2d

		# Kills all python processes:
		nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

		echo "Stopped training of exp type: ${ExpT[$i]}, with it_i: ${ExpI}, Unet"
	else
		# Experiment that I want to run
		CUDA_VISIBLE_DEVICES=${GPU_i} python exp_Unet.py -p with exp_type=${ExpT[$i} it_i=${ExpI} 

		echo "Stopped training of exp type: ${ExpT[$i]}, with it_i: ${ExpI[$i]}, Unet"
	fi
done


