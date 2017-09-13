# -*- coding: iso-8859-1 -*-
# Maintainer: csadorf

from hoomd import *
from hoomd import md
import hoomd;
context.initialize()
import unittest
import os
import tempfile

# unit tests for meta.dump_metadata
class metadata_tests(unittest.TestCase):

    def setUp(self):
        print()
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

    def tearDown(self):
        if init.is_initialized():
            context.initialize()

    def test_with_simulation_run(self):
        nl = md.nlist.cell()
        lj = md.pair.lj(r_cut=3.0, nlist=nl)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        md.integrate.mode_standard(dt=0.01)
        md.integrate.nvt(group=hoomd.group.all(), kT=1.0, tau=1.0)
        hoomd.run(10)
        hoomd.meta.dump_metadata()
        hoomd.run(10)
        hoomd.meta.dump_metadata()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
