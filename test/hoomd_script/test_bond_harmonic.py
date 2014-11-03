# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests bond.harmonic
class bond_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = data.boxdim(L=35);
        self.separation=dict(A=0.35, B=0.35)
        self.s=init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);

        sorter.set_params(grid=8)

    # test to see that se can create a force.constant
    def test_create(self):
        bond.harmonic();

    # test setting coefficients
    def test_set_coeff(self):
        harmonic = bond.harmonic();
        harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = bond.harmonic();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    # test remove particle fails
    def test_bond_fail(self):
        harmonic = bond.harmonic();
        harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        # remove a particle
        del(self.s.particles[0])
        if comm.get_num_ranks() == 1:
            self.assertRaises(RuntimeError, run, 100);
        else:
            # in MPI simulations, we cannot check for an assertion during a simulation
            # the program will terminate with MPI_Abort
            #self.assertRaises(RuntimeError, run, 100);
            pass

    # test adding a dimer
    def test_add_dimer(self):
        harmonic = bond.harmonic();
        harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        t0 = self.s.particles.add('A')
        t1 = self.s.particles.add('B')
        self.s.bonds.add('polymer',t0,t1)
        run(100)

    def tearDown(self):
        del self.s
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
