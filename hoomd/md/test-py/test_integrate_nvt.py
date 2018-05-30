# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# unit tests for md.integrate.nvt
class integrate_nvt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        md.force.constant(fx=0.1, fy=0.1, fz=0.1)

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nvt(all, kT=1.2, tau=0.5);
        run(100);

    # test set_params
    def test_set_params(self):
        all = group.all();
        nvt = md.integrate.nvt(all, kT=1.2, tau=0.5);
        nvt.set_params(kT=1.3);
        nvt.set_params(tau=0.6);

    # test re-initialization of integrator variables
    def test_reinit(self):
        all = group.all()
        integrator = md.integrate.mode_standard(dt=0.005);
        nvt = md.integrate.nvt(all, kT=1.0, tau=0.5)
        integrator.reset_methods()

    # test w/ empty group
    def test_empty(self):
        # currently cannot catch run-time errors in MPI simulations
        pass

        #empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        #mode = md.integrate.mode_standard(dt=0.005);
        #with self.assertRaises(RuntimeError):
        #    nvt = md.integrate.nvt(group=empty, kT=1.0, tau=0.5)
        #    run(1);

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
