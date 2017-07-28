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
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        dpdc = md.pair.dpd_conservative(r_cut=3.0, nlist = self.nl);
        dpdc.pair_coeff.set('A', 'A', A=1.0, r_cut=2.5);
        dpdc.update_coeffs();

    # test missing coefficients
    def test_set_missing_A(self):
        dpdc = md.pair.dpd_conservative(r_cut=3.0, nlist = self.nl);
        dpdc.pair_coeff.set('A', 'A', r_cut=1.0);
        self.assertRaises(RuntimeError, dpdc.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        dpdc = md.pair.dpd_conservative(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, dpdc.update_coeffs);

    def tearDown(self):
        del self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
