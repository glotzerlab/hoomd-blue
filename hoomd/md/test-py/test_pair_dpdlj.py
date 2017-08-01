# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.dpdlj
class pair_dpd_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        dpdlj = md.pair.dpdlj(r_cut=2.5, nlist=self.nl, kT=1.0, seed=10);
        dpdlj.pair_coeff.set('A', 'A', epsilon=1.0, sigma = 1.0, gamma = 4.5);
        dpdlj.update_coeffs();

    # test missing coefficients
    def test_set_missing_gamma(self):
        dpdlj = md.pair.dpdlj(r_cut=2.5, nlist=self.nl, kT=1.0, seed=10);
        dpdlj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        self.assertRaises(RuntimeError, dpdlj.update_coeffs);

    # test missing coefficients
    def test_set_missing_sigma(self):
        dpdlj = md.pair.dpdlj(r_cut=2.5, nlist=self.nl, kT=1.0, seed=10);
        dpdlj.pair_coeff.set('A', 'A', epsilon=1.0, gamma=4.5);
        self.assertRaises(RuntimeError, dpdlj.update_coeffs);

    # test missing coefficients
    def test_set_missing_epsilon(self):
        dpdlj = md.pair.dpdlj(r_cut=2.5, nlist=self.nl, kT=1.0, seed=10);
        dpdlj.pair_coeff.set('A', 'A', sigma=1.0, gamma=4.5);
        self.assertRaises(RuntimeError, dpdlj.update_coeffs);

    # test set_params
    def test(self):
        dpdlj = md.pair.dpdlj(r_cut=2.5, nlist=self.nl, kT=1.0, seed=10);
        dpdlj.pair_coeff.set('A', 'A', epsilon=1.0, sigma = 1.0, gamma = 4.5);
        dpdlj.update_coeffs();
        dpdlj.set_params(kT = 2.0);

    def tearDown(self):
        del self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
