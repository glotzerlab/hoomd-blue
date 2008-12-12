#! /bin/bash

cd pair_lj
qsub $1

cd ../npt
qsub $1

cd ../nve
qsub $1

cd ../bdnvt
qsub $1

cd ../rescale_temp
qsub $1

cd ../bond_fene
qsub $1

cd ../bond_harmonic
qsub $1

cd ../wall_lj
qsub $1
