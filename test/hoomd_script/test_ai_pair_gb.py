# -*- coding: iso-8859-1 -*-
# Maintainer: mspells

from hoomd_script import *
import unittest
import os

# ai_pair.gb
class ai_pair_gb_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        gb = ai_pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0, lpar=1.5);
        gb.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        gb = ai_pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', lperp=1.0, lpar=1.5);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_set_missing_lperp(self):
        gb = ai_pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lpar=1.5);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_set_missing_lpar(self):
        gb = ai_pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        gb = ai_pair.gb(r_cut=3.0);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test set params
    def test_set_params(self):
        gb = ai_pair.gb(r_cut=3.0);
        gb.set_params(mode="no_shift");
        gb.set_params(mode="shift");
        # xplor is not implemented for anisotropic pair potentials
        self.assertRaises(RuntimeError, gb.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_subscribe(self):
        gb = ai_pair.gb(r_cut=2.5);
        gb.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut.get_pair('A','A'));

        gb.pair_coeff.set('A', 'A', r_cut = 2.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, globals.neighbor_list.r_cut.get_pair('A','A'));

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
