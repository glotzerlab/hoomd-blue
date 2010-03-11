# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.slj
class pair_slj_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)
        
    # basic test of creation
    def test(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, r_cut=2.5);
        lj.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        lj.pair_coeff.set('A', 'A', sigma=1.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);
    
    # test set params
    def test_set_params(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        lj.set_params(mode="no_shift");
        lj.set_params(mode="shift");
        self.assertRaises(RuntimeError, lj.set_params, mode="xplor");
        self.assertRaises(RuntimeError, lj.set_params, mode="blah");
    
    # test default coefficients
    def test_default_coeff(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        # (r_cut, and r_on are default)
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        lj.update_coeffs()
    
    # test nlist subscribe
    def teat_nlist_subscribe(self):
        lj = pair.lj(r_cut=2.5, d_max=2.0);
        lj.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(3.5, globals.neighbor_list.r_cut);
        
        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(3.0, globals.neighbor_list.r_cut);
        
        lj.set_params(d_max=3.0);
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(4.0, globals.neighbor_list.r_cut);
        
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

