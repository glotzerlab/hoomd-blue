#!/bin/bash

# This script is a SGE job script for running on teslahoomd.physics.iastate.edu

#$ -N validation
#$ -l gpu=1
#$ -j y
#$ -cwd
#$ -S /bin/bash

source ~/.bashrc

# determine which GPU to run on
GPU=`/home/joaander/gputop/gputop.py --reserve`
# check if no free GPU was found
if [ "$?" = 1 ]; then
	echo "Error finding free GPU"
	echo $GPU
	exit 100
fi

echo "Running hoomd on gpu $GPU"
./run_all.sh --mode=gpu --gpu=$GPU
