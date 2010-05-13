#! /usr/bin/env hoomd
# $Id$
# $URL$

from hoomd_script import *
import math

# parameters
phi_P = 0.20
n_poly = 2371
T = 1.2
polymer1 = dict(bond_len=1.2, type=['A']*10 + ['B']*7 + ['A']*10, bond="linear", count=n_poly)

# perform some simple math to find the length of the box
N = len(polymer1['type']) * polymer1['count'];
L = math.pow(math.pi * N / (6.0 * phi_P), 1.0/3.0);

# generate the polymer system
init.create_random_polymers(box=hoomd.BoxDim(L), polymers=[polymer1], separation=dict(A=0.35, B=0.35));

# force field setup
harmonic = bond.harmonic()
harmonic.set_coeff('polymer', k=330.0, r0=0.84)
lj = pair.lj(r_cut=3.0)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=0.0)
lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, alpha=0.0)
lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, alpha=1.0)

# NVT integration
all = group.all()
integrate.mode_standard(dt=0.005)
nvt = integrate.nvt(group=all, T=T, tau=0.5)

run(10000)

nvt.disable()
bdnvt = integrate.bdnvt(group=all, T=T, tally=True)
analyze.log(filename='bdnvt.log', quantities=['potential_energy', 'kinetic_energy', 'temperature', 'bdnvt_reservoir_energy_all'], period=100, overwrite=True)

run(10000)
