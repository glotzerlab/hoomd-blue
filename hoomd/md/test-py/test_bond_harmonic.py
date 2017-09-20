# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# tests md.bond.harmonic
class bond_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.read_gsd(os.path.join(os.path.dirname(__file__),'test_data_polymer_system.gsd'));
        context.current.sorter.set_params(grid=8)

    # test to see that se can create a md.force.constant
    def test_create(self):
        md.bond.harmonic();

    # test setting coefficients
    def test_set_coeff(self):
        harmonic = md.bond.harmonic();
        harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = md.bond.harmonic();
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    # test remove particle fails
    def test_bond_fail(self):
        harmonic = md.bond.harmonic();
        harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
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
        harmonic = md.bond.harmonic();
        harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        t0 = self.s.particles.add('A')
        t1 = self.s.particles.add('B')
        self.s.bonds.add('polymer',t0,t1)
        run(100)

    # test exclusions in neighbor list
    def test_exclusions(self):
        harmonic = md.bond.harmonic();
        harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        nl = md.nlist.cell()
        lj = md.pair.lj(r_cut=3.0, nlist = nl)
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0);
        lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0);
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100)

        self.assertEqual(nl.cpp_nlist.getNumExclusions(2), (17*100+2*10))
        self.assertEqual(nl.cpp_nlist.getNumExclusions(1), (2*100+2*10))

        # delete bonds connected to a particle
        tags = []
        for b in self.s.bonds:
            if b.a == 2 or b.b == 2:
                tags.append(b.tag)

        for t in tags:
            self.s.bonds.remove(t)

        # delete particle
        self.s.particles.remove(2)

        run(100)

        self.assertEqual(nl.cpp_nlist.getNumExclusions(2), (17*100+2*10)-3)
        self.assertEqual(nl.cpp_nlist.getNumExclusions(1), (2*100+2*10)+2)
        del nl
        del lj
        del harmonic

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
