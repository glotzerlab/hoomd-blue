#! /usr/bin/env hoomd

from hoomd_script import *

init.create_random(N=10000, phi_p=0.2)
lj = pair.lj(r_cut=2.5)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)

all = group.all()
integrate.mode_standard(dt=0.005)
integrate.nvt(group=all, T=1.2, tau=0.5)

analyze.log(filename = 'thermo.log', quantities = ['potential_energy', 'kinetic_energy', 'nvt_reservoir_energy','momentum'], period=1000) 

run(100e6)

