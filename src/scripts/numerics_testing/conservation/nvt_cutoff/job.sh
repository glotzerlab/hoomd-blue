#!/bin/bash

#$ -N nvt_cut
#$ -l gpu=1
#$ -j y
#$ -cwd
#$ -S /bin/bash
#$ -t 1-21

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
hoomd run.hoomd --mode=gpu --gpu=$GPU
