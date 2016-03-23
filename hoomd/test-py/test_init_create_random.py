# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# unit tests for init.create_random
class init_create_random_tests (unittest.TestCase):
    def setUp(self):
        print

    # tests basic creation of the random initializer
    def test(self):
        init.create_random(N=100, phi_p=0.05);
        self.assert_(hoomd_script.context.current.system_definition);
        self.assert_(hoomd_script.context.current.system);

    # tests creation with a few more arguments specified
    def test_moreargs(self):
        init.create_random(name="B", min_dist=0.1, N=100, phi_p=0.05);
        self.assert_(hoomd_script.context.current.system_definition);
        self.assert_(hoomd_script.context.current.system);

    # tests creation with a specified box
    def test_box(self):
        init.create_random(name="B", min_dist=0.1, N=100, box=data.boxdim(L=100));
        self.assert_(hoomd_script.context.current.system_definition);
        self.assert_(hoomd_script.context.current.system);

    # checks for an error if initialized twice
    def test_inittwice(self):
        init.create_random(N=100, phi_p=0.05);
        self.assertRaises(RuntimeError, init.create_random, N=100, phi_p=0.05);

    # test that angle,dihedral, and improper types are initialized correctly
    def test_angleA(self):
        s = init.create_random(N=100, phi_p=0.05);
        snap = s.take_snapshot(all=True);
        self.assertEqual(len(snap.bonds.types), 0);
        self.assertEqual(len(snap.impropers.types), 0);
        self.assertEqual(len(snap.angles.types), 0);
        self.assertEqual(len(snap.dihedrals.types), 0);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
