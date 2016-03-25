# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.nlist testing
class pair_nlist_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=1000, phi_p=0.05);
        #indirectly create the global neighbor list by creating a md.pair.lj without an argument
        md.pair.lj(r_cut=3.0);

        sorter.set_params(grid=8)

    # test set_params via the wrapper
    def test_set_params(self):
        md.nlist.set_params(r_buff=0.6);
        md.nlist.set_params(check_period = 20);
        md.nlist.set_params(d_max = 2.0, dist_check = False)

    # test reset_exclusions via the wrapper
    def test_reset_exclusions_works(self):
        md.nlist.reset_exclusions();
        md.nlist.reset_exclusions(exclusions = ['1-2']);
        md.nlist.reset_exclusions(exclusions = ['1-3']);
        md.nlist.reset_exclusions(exclusions = ['1-4']);
        md.nlist.reset_exclusions(exclusions = ['bond']);
        md.nlist.reset_exclusions(exclusions = ['angle']);
        md.nlist.reset_exclusions(exclusions = ['dihedral']);
        md.nlist.reset_exclusions(exclusions = ['bond', 'angle']);

    # test reset_exclusions error messages
    def test_reset_exclusions_nowork(self):
        self.assertRaises(RuntimeError,
                          md.nlist.reset_exclusions,
                          exclusions = ['bond', 'angle', 'invalid']);

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
