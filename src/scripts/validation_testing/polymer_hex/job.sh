#!/bin/bash

# This script is a SGE job script for running on teslahoomd.physics.iastate.edu

#$ -N nrg_longrun
#$ -l gpu=1
#$ -j y
#$ -cwd
#$ -S /bin/bash

# determine which GPU to run on
GPU=`/home/joaander/gputop/gputop.py --reserve`
# check if no free GPU was found
if [ "$?" = 1 ]; then
	echo "Error finding free GPU"
	echo $GPU
	exit 100
fi

echo "Running hoomd on gpu $GPU"
export PATH=$PATH:/home/joaander/hoomd/bin_rel/python
hoomd polymer_hex.hoomd --mode=gpu --gpu=$GPU
