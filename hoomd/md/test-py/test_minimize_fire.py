# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# unit tests for md.integrate.mode_minimize_fire()
class minimize_fire_tests (unittest.TestCase):
    def setUp(self):
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4])
        md.force.constant(fx=0.1, fy=0.1, fz=0.1)

        context.current.sorter.set_params(grid=8)

    # tests creation of the method with options
    def test(self):
        all = group.all();
        md.integrate.mode_minimize_fire(dt=0.005,Nmin=5, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol = 1e-1, Etol= 1e-5, min_steps=10);
        md.integrate.nve(all)
        run(100);

    # tests with box relaxation
    def test_box_relax(self):
        all = group.all();
        md.integrate.mode_minimize_fire(dt=0.005,Nmin=5, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol = 1e-1, Etol= 1e-5, min_steps=10);
        md.integrate.nph(group=all, gamma=1,P=0.0,tauP=1.0)
        run(100);

    # tests anisotropic option
    def test_aniso(self):
        all = group.all();
        fire = md.integrate.mode_minimize_fire(dt=0.005,aniso=True)
        md.integrate.nve(all)
        run(100);
        fire.set_params(aniso=False)
        run(100);

    # test w/ empty group
    def test_empty(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = md.integrate.mode_minimize_fire(dt=0.005);
        nve = md.integrate.nve(group=empty)
        run(1);

    # test method can be enabled and disabled
    def test_disable_enable(self):
        mode = md.integrate.mode_minimize_fire(dt=0.005);
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
