#! /bin/bash

cd pair_lj
echo "****************"
echo "Running pair_lj"
hoomd run.hoomd "$@"

cd ../npt
echo "****************"
echo "Running npt"
hoomd run.hoomd "$@"

cd ../nve
echo "****************"
echo "Running nve"
hoomd run.hoomd "$@"

cd ../bdnvt
echo "****************"
echo "Running bdnvt"
hoomd run.hoomd "$@"

cd ../rescale_temp
echo "****************"
echo "Running rescale_temp"
hoomd run.hoomd "$@"

cd ../bond_fene
echo "****************"
echo "Running bond_fene"
hoomd run.hoomd "$@"

cd ../bond_harmonic
echo "****************"
echo "Running bond_harmonic"
hoomd run.hoomd "$@"

cd ../wall_lj
echo "****************"
echo "Running wall_lj"
hoomd run.hoomd "$@"

