# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.yukawa
class pair_morse_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)
        
    # basic test of creation
    def test(self):
        p = pair.morse(r_cut=3.0);
        p.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0, r_cut=3.0, r_on=2.0);
        p.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        p = pair.morse(r_cut=3.0);
        p.pair_coeff.set('A', 'A', D0=1.0);
        self.assertRaises(RuntimeError, p.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        p = pair.morse(r_cut=3.0);
        self.assertRaises(RuntimeError, p.update_coeffs);
    
    # test set params
    def test_set_params(self):
        p = pair.morse(r_cut=3.0);
        p.set_params(mode="no_shift");
        p.set_params(mode="shift");
        p.set_params(mode="xplor");
        self.assertRaises(RuntimeError, p.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_subscribe(self):
        p = pair.morse(r_cut=2.5);
        p.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut);
        
        p.pair_coeff.set('A', 'A', r_cut = 2.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, globals.neighbor_list.r_cut);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

