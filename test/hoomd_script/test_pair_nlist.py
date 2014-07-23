# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.nlist testing
class pair_nlist_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        #indirectly create the neighbor list by creating a pair.lj
        pair.lj(r_cut=3.0);

        sorter.set_params(grid=8)


    # test set_params
    def test_set_params(self):
        globals.neighbor_list.set_params(r_buff=0.6);
        globals.neighbor_list.set_params(check_period = 20);
        globals.neighbor_list.set_params(deterministic = True);

    # test reset_exclusions
    def test_reset_exclusions_works(self):
        globals.neighbor_list.reset_exclusions();
        globals.neighbor_list.reset_exclusions(exclusions = ['1-2']);
        globals.neighbor_list.reset_exclusions(exclusions = ['1-3']);
        globals.neighbor_list.reset_exclusions(exclusions = ['1-4']);
        globals.neighbor_list.reset_exclusions(exclusions = ['bond']);
        globals.neighbor_list.reset_exclusions(exclusions = ['angle']);
        globals.neighbor_list.reset_exclusions(exclusions = ['dihedral']);
        globals.neighbor_list.reset_exclusions(exclusions = ['bond', 'angle']);

    # test reset_exclusions error messages
    def test_reset_exclusions_nowork(self):
        self.assertRaises(RuntimeError,
                          globals.neighbor_list.reset_exclusions,
                          exclusions = ['bond', 'angle', 'invalid']);

    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
