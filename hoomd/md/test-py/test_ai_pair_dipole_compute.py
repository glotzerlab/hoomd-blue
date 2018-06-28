# -*- coding: iso-8859-1 -*-
# Maintainer: mspells

import math as m
from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy

class pair_dipole_tests_force (unittest.TestCase):
    def setUp(self):
        print
        context.initialize()

        snapshot = data.make_snapshot(N=2, box=data.boxdim(L=1000.0))
        if comm.get_rank() == 0:
            snapshot.particles.position[0] = [0.0, 0.0, 0.0]
            snapshot.particles.position[1] = [0.8, 0.45, 0.9]
            snapshot.particles.orientation[0] = [1, 0, 0, 0]
            snapshot.particles.orientation[1] = [m.cos(2*m.pi/6), m.sin(2*m.pi/6)/m.sqrt(2), m.sin(2*m.pi/6)/m.sqrt(2), 0]
            snapshot.particles.charge[0] = 2.0
            snapshot.particles.charge[1] = 1.0
        self.system = init.read_snapshot(snapshot)

    def test_force_torque_kappa0(self):

        self.nl = md.nlist.cell()
        dipole = md.pair.dipole(r_cut=6.0, nlist = self.nl)
        dipole.pair_coeff.set('A', 'A', mu=0.6, A=1.0, kappa=0.0)
        md.integrate.mode_standard(dt = 0.0)
        md.integrate.nve(group = group.all())
        run(1)

        force_1 = self.system.particles[0].net_force
        torque_1 = self.system.particles[0].net_torque
        numpy.testing.assert_allclose(-1.0783202, force_1[0], rtol=1e-6)
        numpy.testing.assert_allclose(-1.2620099, force_1[1], rtol=1e-6)
        numpy.testing.assert_allclose(-0.8108346, force_1[2], rtol=1e-6)
        numpy.testing.assert_allclose(0, torque_1[0], atol=1e-7)
        numpy.testing.assert_allclose(0.1542009, torque_1[1], rtol=1e-6)
        numpy.testing.assert_allclose(-0.2560913, torque_1[2], rtol=1e-6)

        force_2 = self.system.particles[1].net_force
        torque_2 = self.system.particles[1].net_torque
        numpy.testing.assert_allclose(1.0783202, force_2[0], rtol=1e-6)
        numpy.testing.assert_allclose(1.2620099, force_2[1], rtol=1e-6)
        numpy.testing.assert_allclose(0.8108346, force_2[2], rtol=1e-6)
        numpy.testing.assert_allclose(0.7709333, torque_2[0], rtol=1e-6)
        numpy.testing.assert_allclose(-0.4760214, torque_2[1], rtol=1e-6)
        numpy.testing.assert_allclose(-0.2682726, torque_2[2], rtol=1e-6)

    def test_force_torque_kappa1(self):

        self.nl = md.nlist.cell()
        dipole = md.pair.dipole(r_cut=6.0, nlist = self.nl)
        dipole.pair_coeff.set('A', 'A', mu=0.6, A=1.0, kappa=1.0)
        md.integrate.mode_standard(dt = 0.0)
        md.integrate.nve(group = group.all())
        run(1)

        force_1 = self.system.particles[0].net_force
        torque_1 = self.system.particles[0].net_torque
        numpy.testing.assert_allclose(-0.6139756, force_1[0], rtol=1e-6)
        numpy.testing.assert_allclose(-0.5266033, force_1[1], rtol=1e-6)
        numpy.testing.assert_allclose(-0.5794879, force_1[2], rtol=1e-6)
        numpy.testing.assert_allclose(0, torque_1[0], atol=1e-7)
        numpy.testing.assert_allclose(0.0426386, torque_1[1], rtol=1e-6)
        numpy.testing.assert_allclose(-0.0708127, torque_1[2], rtol=1e-6)

        force_2 = self.system.particles[1].net_force
        torque_2 = self.system.particles[1].net_torque
        numpy.testing.assert_allclose(0.6139756, force_2[0], rtol=1e-6)
        numpy.testing.assert_allclose(0.5266033, force_2[1], rtol=1e-6)
        numpy.testing.assert_allclose(0.5794879, force_2[2], rtol=1e-6)
        numpy.testing.assert_allclose(0.2131734, torque_2[0], rtol=1e-6)
        numpy.testing.assert_allclose(-0.1316263, torque_2[1], rtol=1e-6)
        numpy.testing.assert_allclose(-0.0741810, torque_2[2], rtol=1e-6)

    def tearDown(self):
        del self.system, self.nl
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
