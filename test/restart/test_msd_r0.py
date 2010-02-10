#! /usr/bin/env hoomd

from hoomd_script import *

init.create_random(N=64000, phi_p=0.2)
lj = pair.lj(r_cut=3.0)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)

all = group.all()
integrate.mode_standard(dt=0.005)
integrate.nvt(group=all, T=1.2, tau=0.5)

# warm up
run(2000)

# save the current particle state
xml = dump.xml()
xml.set_params(all=True)
xml.write(filename="init.xml")

# run for a bit longer to move the particles
run(2000)

# start two msd analyzers, one without the r0 file and one with r0_file="init.xml"
# they should have two different starting points
all = group.all()
analyze.msd(filename="msd_test_nor0.log", groups=[all], period=10)
analyze.msd(filename="msd_test_r0.log", groups=[all], period=10, r0_file='init.xml')

run(2000)
