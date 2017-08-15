# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# unit tests for md.integrate.nve
class integrate_nve_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        md.force.constant(fx=0.1, fy=0.1, fz=0.1)

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # tests creation of the method with options
    def test(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all, limit=0.01, zero_force=True);
        run(100);

    # test set_params
    def test_set_params(self):
        all = group.all();
        mode = md.integrate.mode_standard(dt=0.005);
        mode.set_params(dt=0.001);
        nve = md.integrate.nve(all);
        nve.set_params(limit=False);
        nve.set_params(limit=0.1);
        nve.set_params(zero_force=False);

    # test w/ empty group
    def test_empty(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = md.integrate.mode_standard(dt=0.005);
        nve = md.integrate.nve(group=empty)
        run(1);

    # test method can be enabled and disabled
    def test_disable_enable(self):
        mode = md.integrate.mode_standard(dt=0.005);
        nve = md.integrate.nve(group=group.all())
        self.assertTrue(nve in context.current.integration_methods)

        # disable this integrator, which removes it from the context
        nve.disable()
        self.assertFalse(nve in context.current.integration_methods)
        # second call does nothing
        nve.disable()

        # reenable the integrator
        nve.enable()
        self.assertTrue(nve in context.current.integration_methods)
        # second call does nothing
        nve.enable()

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
