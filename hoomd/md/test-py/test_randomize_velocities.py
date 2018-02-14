# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

# unit tests for velocity randomization
class velocity_randomization_tests (unittest.TestCase):

    def setUp(self):
        #target a packing fraction of 0.05
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4])

    def test_nvt(self):
        all = group.all()
        md.integrate.mode_standard(dt=0.005)
        integrator = md.integrate.nvt(group=all, kT=1.0, tau=0.5)
        integrator.randomize_velocities(kT=1.0, seed=42)
        run(100)

    def test_npt(self):
        all = group.all()
        md.integrate.mode_standard(dt=0.005)
        integrator = md.integrate.npt(group=all, kT=1.0, tau=0.5, tauP=1.0, P=2.0)
        integrator.randomize_velocities(kT=1.0, seed=42)
        run(100)

    def test_nph(self):
        all = group.all()
        md.integrate.mode_standard(dt=0.005)
        integrator = md.integrate.nph(group=all, P=2.0, tauP=1.0)
        integrator.randomize_velocities(kT=1.0, seed=42)
        run(100)

    def test_nve(self):
        all = group.all()
        md.integrate.mode_standard(dt=0.005)
        integrator = md.integrate.nve(group=all)
        integrator.randomize_velocities(kT=1.0, seed=42)
        run(100)

    def tearDown(self):
        context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
