#!/bin/bash

# This script is a SGE job script for running on teslahoomd.physics.iastate.edu

#$ -N validation
#$ -l gpu=1
#$ -j y
#$ -cwd
#$ -S /bin/bash
#$ -t 1-8
#$ -v HOOMD_ARGS

source ~/.bashrc

# determine which GPU to run on
GPU=`/home/joaander/gputop/gputop.py --reserve`
# check if no free GPU was found
if [ "$?" = 1 ]; then
	echo "Error finding free GPU"
	echo $GPU
	exit 1
fi

directory_list=( dummy pair_lj npt nve bdnvt rescale_temp bond_fene bond_harmonic wall_lj )
echo "Running hoomd validation test ${directory_list[${SGE_TASK_ID}]} on gpu ${GPU} with args ${HOOMD_ARGS}"
cd ${directory_list[${SGE_TASK_ID}]}
hoomd run.hoomd ${HOOMD_ARGS} --mode=gpu --gpu=$GPU
