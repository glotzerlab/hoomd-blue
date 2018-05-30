# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# unit tests for init.create_random
class replicate(unittest.TestCase):
    def setUp(self):
        self.s = init.read_gsd(os.path.join(os.path.dirname(__file__),'test_data_polymer_system.gsd'));
        self.assert_(context.current.system_definition);
        self.assert_(context.current.system);
        self.harmonic = md.bond.harmonic();
        self.harmonic.bond_coeff.set('polymer', k=1.0, r0=1.0)
        self.nl = md.nlist.cell()
        self.pair = md.pair.lj(r_cut=2.5, nlist = self.nl)
        self.pair.pair_coeff.set('A','A',epsilon=1.0, sigma=1.0)
        self.pair.pair_coeff.set('A','B',epsilon=1.0, sigma=1.0)
        self.pair.pair_coeff.set('B','B',epsilon=1.0, sigma=1.0)

    def test_replicate(self):
        self.s.replicate(nx=2,ny=2,nz=2)
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(group.all());
        run(100)
        self.assertEqual(len(self.s.particles),8*(19*100+4*10))
        self.assertEqual(len(self.s.bonds),8*(18*100+3*10))
        self.assertEqual(self.nl.cpp_nlist.getNumExclusions(2), 8*(17*100+2*10))
        self.assertEqual(self.nl.cpp_nlist.getNumExclusions(1), 8*(2*100+2*10))
        run(100);

    def tearDown(self):
        del self.harmonic
        del self.pair
        del self.s
        del self.nl
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
