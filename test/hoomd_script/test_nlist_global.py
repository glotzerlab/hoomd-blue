# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# pair.nlist testing
class pair_nlist_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=1000, phi_p=0.05);
        #indirectly create the global neighbor list by creating a pair.lj without an argument
        pair.lj(r_cut=3.0);

        sorter.set_params(grid=8)

    # test set_params via the wrapper
    def test_set_params(self):
        nlist.set_params(r_buff=0.6);
        nlist.set_params(check_period = 20);
        nlist.set_params(d_max = 2.0, dist_check = False)

    # test reset_exclusions via the wrapper
    def test_reset_exclusions_works(self):
        nlist.reset_exclusions();
        nlist.reset_exclusions(exclusions = ['1-2']);
        nlist.reset_exclusions(exclusions = ['1-3']);
        nlist.reset_exclusions(exclusions = ['1-4']);
        nlist.reset_exclusions(exclusions = ['bond']);
        nlist.reset_exclusions(exclusions = ['angle']);
        nlist.reset_exclusions(exclusions = ['dihedral']);
        nlist.reset_exclusions(exclusions = ['bond', 'angle']);

    # test reset_exclusions error messages
    def test_reset_exclusions_nowork(self):
        self.assertRaises(RuntimeError,
                          nlist.reset_exclusions,
                          exclusions = ['bond', 'angle', 'invalid']);

    # test tuning via the wrapper
    def test_tune(self):
        tune.r_buff(warmup=100, r_min=0.1, r_max=0.25, jumps=10, steps=50)

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
