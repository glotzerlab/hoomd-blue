#! /usr/bin/env hoomd

import os.path
from hoomd_script import *

if os.path.isfile('continue.bin.gz'):
    init.read_bin(filename='continue.bin.gz')
else:
    init.create_random(N=64000, phi_p=0.2)
    # save the initial state
    xml = dump.xml()
    xml.set_params(all=True);
    xml.write(filename="init.xml")

lj = pair.lj(r_cut=3.0)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)

all = group.all()
integrate.mode_standard(dt=0.005)
integrate.nvt(group=all, T=1.2, tau=0.5)

# start two msd analyzers, one without the r0 file and one with r0_file="init.xml"
# one should correctly append to the existing file while the other will restart counting at 0
analyze.msd(filename="msd_test_nor0.log", groups=[all], period=10)
analyze.msd(filename="msd_test_r0.log", groups=[all], period=10, r0_file='init.xml')
# start a third to test the file overwrite option
analyze.msd(filename="msd_test_overwrite.log", groups=[all], period=10, overwrite=True)

run(2000)

bin = dump.bin()
bin.write(filename="continue.bin.gz")
