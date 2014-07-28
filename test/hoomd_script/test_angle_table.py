# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os
import math

# tests angle.table
class angle_table_tests (unittest.TestCase):
    def setUp(self):
        print
        # create a polymer system and add a angle to it
        self.polymer1 = dict(bond_len=1.2, type=['A']*3, bond="linear", count=10);
        self.polymers = [self.polymer1]
        self.box = data.boxdim(L=35);
        self.separation=dict(A=0.35, B=0.35)
        self.sys = init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);

        for i in range(len(self.sys.particles)//3-1):
            self.sys.angles.add('angleA', 3*i+0, 3*i+1, 3*i+2);

        sorter.set_params(grid=8)

    # test to see that se can create a angle.table
    def test_create(self):
        angle.table(width=100);

    # test setting the table
    def test_set_coeff(self):
        def har(theta, kappa, theta_0):
            V = 0.5 * kappa * (theta-theta_0)**2;
            T = -kappa*(theta-theta_0);
            return (V, T)

        harmonic = angle.table(width=1000)
        harmonic.angle_coeff.set('angleA', func=har, coeff=dict(kappa=30,theta_0=.1))
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = angle.table(width=123)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    # compare against harmonic angle
    def test_harmonic_compare(self):
        harmonic_1 = angle.table(width=1000)
        harmonic_1.angle_coeff.set('angleA', func=lambda theta: (0.5*1*theta*theta, -theta), coeff=dict())
        harmonic_2 = angle.harmonic()
        harmonic_2.set_coeff('angleA', k=1.0, t0=0)
        integrate.mode_standard(dt=0.005);
        all = group.all()
        integrate.nve(all)
        run(1)
        for i in range(len(self.sys.particles)):
            f_1 = harmonic_1.forces[i]
            f_2 = harmonic_2.forces[i]
            # we have to have a very rough tolerance (~10%) because
            # of 1) discretization of the potential and 2) different handling of precision issues in both potentials
            self.assertAlmostEqual(f_1.energy, f_2.energy,3)
            self.assertAlmostEqual(f_1.force[0], f_2.force[0],2)
            self.assertAlmostEqual(f_1.force[1], f_2.force[1],2)
            self.assertAlmostEqual(f_1.force[2], f_2.force[2],2)

    def tearDown(self):
        del self.sys
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
