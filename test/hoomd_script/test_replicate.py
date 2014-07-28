# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random
class replicate(unittest.TestCase):
    def setUp(self):
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = data.boxdim(L=35);
        self.separation=dict(A=0.42, B=0.42)
        self.s = init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.harmonic = bond.harmonic();
        self.harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        self.pair = pair.lj(r_cut=2.5)
        self.pair.pair_coeff.set('A','A',epsilon=1.0, sigma=1.0)
        self.pair.pair_coeff.set('A','B',epsilon=1.0, sigma=1.0)
        self.pair.pair_coeff.set('B','B',epsilon=1.0, sigma=1.0)

    def test_replicate(self):
        self.s.replicate(nx=2,ny=2,nz=2)
        integrate.mode_standard(dt=0.005);
        integrate.nve(group.all());
        run(100)
        self.assertEqual(len(self.s.particles),8*(19*100+4*10))
        self.assertEqual(len(self.s.bonds),8*(18*100+3*10))
        self.assertEqual(globals.neighbor_list.cpp_nlist.getNumExclusions(2), 8*(17*100+2*10))
        self.assertEqual(globals.neighbor_list.cpp_nlist.getNumExclusions(1), 8*(2*100+2*10))
        run(100);

    def tearDown(self):
        del self.harmonic
        del self.pair
        del self.s
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
