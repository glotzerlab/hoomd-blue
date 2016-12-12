# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import math
import numpy

numpy.random.seed(10)

# tests md.angle.table
class angle_table_tests (unittest.TestCase):
    def setUp(self):
        print
        snap = data.make_snapshot(N=40,
                                  box=data.boxdim(L=100),
                                  particle_types = ['A'],
                                  bond_types = [],
                                  angle_types = ['angleA'],
                                  dihedral_types = [],
                                  improper_types = [])

        if comm.get_rank() == 0:
            snap.angles.resize(10);

            for i in range(10):
                x = numpy.array([i, 0, 0], dtype=numpy.float32)
                snap.particles.position[4*i+0,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+1,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+2,:] = x;
                x += numpy.random.random(3)
                snap.particles.position[4*i+3,:] = x;

                snap.angles.group[i,:] = [4*i+0, 4*i+1, 4*i+2];

        self.sys = init.read_snapshot(snap)

        context.current.sorter.set_params(grid=8)

    # test to see that se can create a md.angle.table
    def test_create(self):
        md.angle.table(width=100);

    # test setting the table
    def test_set_coeff(self):
        def har(theta, kappa, theta_0):
            V = 0.5 * kappa * (theta-theta_0)**2;
            T = -kappa*(theta-theta_0);
            return (V, T)

        harmonic = md.angle.table(width=1000)
        harmonic.angle_coeff.set('angleA', func=har, coeff=dict(kappa=30,theta_0=.1))
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = md.angle.table(width=123)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    # compare against harmonic angle
    def test_harmonic_compare(self):
        harmonic_1 = md.angle.table(width=1000)
        harmonic_1.angle_coeff.set('angleA', func=lambda theta: (0.5*1*theta*theta, -theta), coeff=dict())
        harmonic_2 = md.angle.harmonic()
        harmonic_2.angle_coeff.set('angleA', k=1.0, t0=0)
        md.integrate.mode_standard(dt=0.005);
        all = group.all()
        md.integrate.nve(all)
        run(1)
        for i in range(len(self.sys.particles)):
            f_1 = harmonic_1.forces[i]
            f_2 = harmonic_2.forces[i]
            # we have to have a very rough tolerance (~10%) because
            # of 1) discretization of the potential and 2) different handling of precision issues in both potentials
            numpy.testing.assert_allclose(f_1.energy, f_2.energy,rtol=0.001)
            numpy.testing.assert_allclose(f_1.force[0], f_2.force[0],rtol=0.01)
            numpy.testing.assert_allclose(f_1.force[1], f_2.force[1],rtol=0.01)
            numpy.testing.assert_allclose(f_1.force[2], f_2.force[2],rtol=0.01)

    def tearDown(self):
        del self.sys
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
