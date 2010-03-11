# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random_polymers
class init_create_random_polymer_tests (unittest.TestCase):
    def setUp(self):
        print
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        import __main__;
        __main__.sorter.set_params(grid=8)
    
    # tests basic creation of the random initializer
    def test(self):
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
    
    # checks for an error if initialized twice
    def test_create_random_inittwice(self):
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        self.assertRaises(RuntimeError, 
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=self.separation);
    
    # checks that invalid arguments are detected
    def test_bad_polymers(self):
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymer1,
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=5,
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers="polymers",
                          separation=self.separation);
        
        bad_polymer1 = dict(bond_len=1.2, bond="linear", count=10)
        bad_polymer2 = dict(type=['B']*4, bond="linear", count=10)
        bad_polymer3 = dict(bond_len=1.2, type=['B']*4, count=10)
        bad_polymer4 = dict(bond_len=1.2, type=['B']*4, bond="linear")
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer1],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer2],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer3],
                          separation=self.separation);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=[bad_polymer4],
                          separation=self.separation);
        
    def test_bad_separation(self):
        bad_separation1 = dict(A=0.35)
        bad_separation2 = dict(B=0.35)
        bad_separation3 = dict(C=0.35)
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=bad_separation1);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=bad_separation2);
        self.assertRaises(RuntimeError,
                          init.create_random_polymers,
                          box=self.box,
                          polymers=self.polymers,
                          separation=bad_separation3);
        
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

