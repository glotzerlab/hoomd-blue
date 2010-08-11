#! /usr/bin/env hoomd

# $Id$
# $URL$

from hoomd_script import *

init.create_random(N=64000, phi_p=0.2)
lj = pair.lj(r_cut=3.0)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)

all = group.all()
integrate.mode_standard(dt=0.005)
integrate.nvt(group=all, T=1.2, tau=0.5)

analyze.log(filename='test_thermo.log', quantities=['temperature', 'pressure', 'kinetic_energy', 'potential_energy', 'num_particles', 'ndof'], period=10)

run(1000)

