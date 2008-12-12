#! /bin/bash

cd pair_lj
hoomd run.hoomd "$@"

cd ../npt
hoomd run.hoomd "$@"

cd ../nve
hoomd run.hoomd "$@"

cd ../bdnvt
hoomd run.hoomd "$@"

cd ../rescale_temp
hoomd run.hoomd "$@"

cd ../bond_fene
hoomd run.hoomd "$@"

cd ../bond_harmonic
hoomd run.hoomd "$@"

cd ../wall_lj
hoomd run.hoomd "$@"

