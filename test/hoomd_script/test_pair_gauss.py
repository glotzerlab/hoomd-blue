# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.gauss
class pair_gauss_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)
        
    # basic test of creation
    def test(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        gauss.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.pair_coeff.set('A', 'A', sigma=1.0);
        self.assertRaises(RuntimeError, gauss.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        gauss = pair.gauss(r_cut=3.0);
        self.assertRaises(RuntimeError, gauss.update_coeffs);
    
    # test set params
    def test_set_params(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.set_params(mode="no_shift");
        gauss.set_params(mode="shift");
        gauss.set_params(mode="xplor");
        self.assertRaises(RuntimeError, gauss.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_subscribe(self):
        gauss = pair.gauss(r_cut=2.5);
        gauss.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut);
        
        gauss.pair_coeff.set('A', 'A', r_cut = 2.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, globals.neighbor_list.r_cut);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

