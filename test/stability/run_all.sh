#! /bin/bash

cd pair_lj
echo "****************"
echo "Running pair_lj"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../npt
echo "****************"
echo "Running npt"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../nve
echo "****************"
echo "Running nve"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../bdnvt
echo "****************"
echo "Running bdnvt"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../rescale_temp
echo "****************"
echo "Running rescale_temp"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../bond_fene
echo "****************"
echo "Running bond_fene"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../bond_harmonic
echo "****************"
echo "Running bond_harmonic"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../wall_lj
echo "****************"
echo "Running wall_lj"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../pair_gaussian
echo "****************"
echo "Running pair_gaussian"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../pair_cgcmm
echo "****************"
echo "Running pair_cgcmm"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../angle_harmonic
echo "****************"
echo "Running angle_harmonic"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../pair_slj
echo "****************"
echo "Running pair_slj"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../pair_table
echo "****************"
echo "Running pair_table"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../pair_yukawa
echo "****************"
echo "Running pair_yukawa"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1

cd ../pair_morse
echo "****************"
echo "Running pair_morse"
hoomd run.hoomd "$@" --gpu_error_checking 2>&1
