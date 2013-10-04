# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.dpd
class pair_dpd_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        dpd = pair.dpd(r_cut=3.0, T=1.0);
        dpd.pair_coeff.set('A', 'A', A=1.0, gamma = 4.5, r_cut=2.5);
        dpd.update_coeffs();

    # test missing coefficients
    def test_set_missing_gamma(self):
        dpd = pair.dpd(r_cut=3.0, T=1.0);
        dpd.pair_coeff.set('A', 'A', A=40);
        self.assertRaises(RuntimeError, dpd.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        dpd = pair.dpd(r_cut=3.0, T=1.0);
        self.assertRaises(RuntimeError, dpd.update_coeffs);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
