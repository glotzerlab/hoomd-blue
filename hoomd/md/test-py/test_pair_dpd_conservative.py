# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.dpd
class pair_dpdc_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        dpdc = md.pair.dpd_conservative(r_cut=3.0);
        dpdc.pair_coeff.set('A', 'A', A=1.0, r_cut=2.5);
        dpdc.update_coeffs();

    # test missing coefficients
    def test_set_missing_A(self):
        dpdc = md.pair.dpd_conservative(r_cut=3.0);
        dpdc.pair_coeff.set('A', 'A', r_cut=1.0);
        self.assertRaises(RuntimeError, dpdc.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        dpdc = md.pair.dpd_conservative(r_cut=3.0);
        self.assertRaises(RuntimeError, dpdc.update_coeffs);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
