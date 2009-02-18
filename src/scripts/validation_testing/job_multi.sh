#!/bin/bash

# This script is a SGE job script for running on teslahoomd.physics.iastate.edu

#$ -N validation_multi
#$ -l gpu=2
#$ -j y
#$ -cwd
#$ -S /bin/bash
#$ -t 1-9
#$ -v HOOMD_ARGS

source ~/.bashrc

# determine which GPU to run on
GPU1=`/home/joaander/gputop/gputop.py --reserve`
# check if no free GPU was found
if [ "$?" = 1 ]; then
	echo "Error finding free GPU"
	echo $GPU
	exit 1
fi
GPU2=`/home/joaander/gputop/gputop.py --reserve`
# check if no free GPU was found
if [ "$?" = 1 ]; then
	echo "Error finding free GPU"
	echo $GPU
	exit 100
fi

directory_list=( dummy pair_lj npt nve bdnvt rescale_temp bond_fene bond_harmonic wall_lj pair_gaussian )
echo "Running hoomd validation test ${directory_list[${SGE_TASK_ID}]} on gpu $GPU1,$GPU2 with args ${HOOMD_ARGS}"
cd ${directory_list[${SGE_TASK_ID}]}
hoomd run.hoomd ${HOOMD_ARGS} --gpu_error_checking --mode=gpu --gpu=$GPU1,$GPU2
