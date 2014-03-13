# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random
class init_create_random_tests (unittest.TestCase):
    def setUp(self):
        print

    # tests basic creation of the random initializer
    def test(self):
        init.create_random(N=100, phi_p=0.05);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);

    # tests creation with a few more arguments specified
    def test_moreargs(self):
        init.create_random(name="B", min_dist=0.1, N=100, phi_p=0.05);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);

    # tests creation with a specified box
    def test_box(self):
        init.create_random(name="B", min_dist=0.1, N=100, box=data.boxdim(L=100));
        self.assert_(globals.system_definition);
        self.assert_(globals.system);

    # checks for an error if initialized twice
    def test_inittwice(self):
        init.create_random(N=100, phi_p=0.05);
        self.assertRaises(RuntimeError, init.create_random, N=100, phi_p=0.05);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
