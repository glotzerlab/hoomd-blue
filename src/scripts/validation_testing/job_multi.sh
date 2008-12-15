#!/bin/bash

# This script is a SGE job script for running on teslahoomd.physics.iastate.edu

#$ -N validation_multi
#$ -l gpu=2
#$ -j y
#$ -cwd
#$ -S /bin/bash
#$ -t 1-8
#$ -v HOOMD_ARGS

source ~/.bashrc

# determine which GPU to run on
GPU1=`/home/joaander/gputop/gputop.py --reserve`
# check if no free GPU was found
if [ "$?" = 1 ]; then
	echo "Error finding free GPU"
	echo $GPU
	exit 100
fi
GPU2=`/home/joaander/gputop/gputop.py --reserve`
# check if no free GPU was found
if [ "$?" = 1 ]; then
	echo "Error finding free GPU"
	echo $GPU
	exit 100
fi

echo "Running hoomd on gpu $GPU1,$GPU2 with args ${HOOMD_ARGS}"
directory_list=( pair_lj npt nve bdnvt rescale_temp bond_fene bond_harmonic wall_lj )
cd ${directory_list[${SGE_TASK_ID}]}
hoomd run.hoomd ${HOOMD_ARGS} --mode=gpu --gpu=$GPU1,$GPU2
