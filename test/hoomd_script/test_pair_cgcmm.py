# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.cgcmm
class pair_cgcmm_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        cgcmm = pair.cgcmm(r_cut=3.0);
        cgcmm.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='lj12_4');
        cgcmm.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        cgcmm = pair.cgcmm(r_cut=3.0);
        cgcmm.pair_coeff.set('A', 'A', sigma=1.0, alpha=1.0);
        self.assertRaises(RuntimeError, cgcmm.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        cgcmm = pair.cgcmm(r_cut=3.0);
        self.assertRaises(RuntimeError, cgcmm.update_coeffs);

   # test nlist subscribe
    def test_nlist_subscribe(self):
        cgcmm = pair.cgcmm(r_cut=2.5);
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
