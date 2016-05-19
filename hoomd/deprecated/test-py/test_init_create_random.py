# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
import hoomd;
context.initialize()
import unittest
import os

# unit tests for init.create_random
class init_create_random_tests (unittest.TestCase):
    def setUp(self):
        print

    # tests basic creation of the random initializer
    def test(self):
        deprecated.init.create_random(N=100, phi_p=0.05);
        self.assert_(hoomd.context.current.system_definition);
        self.assert_(hoomd.context.current.system);

    # tests creation with a few more arguments specified
    def test_moreargs(self):
        deprecated.init.create_random(name="B", min_dist=0.1, N=100, phi_p=0.05);
        self.assert_(hoomd.context.current.system_definition);
        self.assert_(hoomd.context.current.system);

    # tests creation with a specified box
    def test_box(self):
        deprecated.init.create_random(name="B", min_dist=0.1, N=100, box=data.boxdim(L=100));
        self.assert_(hoomd.context.current.system_definition);
        self.assert_(hoomd.context.current.system);

    # checks for an error if initialized twice
    def test_inittwice(self):
        deprecated.init.create_random(N=100, phi_p=0.05);
        self.assertRaises(RuntimeError, deprecated.init.create_random, N=100, phi_p=0.05);

    # test that angle,dihedral, and improper types are initialized correctly
    def test_angleA(self):
        s = deprecated.init.create_random(N=100, phi_p=0.05);
        snap = s.take_snapshot(all=True);
        self.assertEqual(len(snap.bonds.types), 0);
        self.assertEqual(len(snap.impropers.types), 0);
        self.assertEqual(len(snap.angles.types), 0);
        self.assertEqual(len(snap.dihedrals.types), 0);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
