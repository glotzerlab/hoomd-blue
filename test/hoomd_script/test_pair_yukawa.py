# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.yukawa
class pair_yukawa_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        
    # basic test of creation
    def test(self):
        yuk = pair.yukawa(r_cut=3.0);
        yuk.pair_coeff.set('A', 'A', epsilon=1.0, kappa=1.0);
        yuk.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        yuk = pair.gauss(r_cut=3.0);
        yuk.pair_coeff.set('A', 'A', kappa=1.0);
        self.assertRaises(RuntimeError, yuk.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        yuk = pair.yukawa(r_cut=3.0);
        self.assertRaises(RuntimeError, yuk.update_coeffs);
    
    # test set params
    def test_set_params(self):
        yuk = pair.gauss(r_cut=3.0);
        yuk.set_params(mode="no_shift");
        yuk.set_params(mode="shift");
        yuk.set_params(mode="xplor");
        self.assertRaises(RuntimeError, yuk.set_params, mode="blah");

    # test nlist subscribe
    def teat_nlist_subscribe(self):
        yuk = pair.yukawa(r_cut=2.5);
        gauss.pair_coeff.set('A', 'A', simga=1.0, kappa=1.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut);
        
        yuk.pair_coeff.set('A', 'A', r_cut = 2.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, globals.neighbor_list.r_cut);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

