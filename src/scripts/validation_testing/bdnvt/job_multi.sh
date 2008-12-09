#!/bin/bash

# This script is a SGE job script for running on teslahoomd.physics.iastate.edu

#$ -N bdnvt_multi
#$ -l gpu=2
#$ -j y
#$ -cwd
#$ -S /bin/bash

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

echo "Running hoomd on gpu $GPU1,$GPU2"
hoomd run.hoomd --mode=gpu --gpu=$GPU1,$GPU2
