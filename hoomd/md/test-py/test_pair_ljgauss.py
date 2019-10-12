# -*- coding: iso-8859-1 -*-
# Maintainer: jproc

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.ljgauss
class pair_ljgauss_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=1.0),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        ljgauss = md.pair.ljgauss(r_cut=3.0, nlist = self.nl);
        ljgauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=0.02**0.5, r0=1.75);
        ljgauss.update_coeffs();

    # test missing coefficients
    def test_set_missing_sigma(self):
        ljgauss = md.pair.ljgauss(r_cut=3.0, nlist = self.nl);
        ljgauss.pair_coeff.set('A', 'A', epsilon=1.0, r0=1.75);
        self.assertRaises(RuntimeError, ljgauss.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        ljgauss = md.pair.ljgauss(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, ljgauss.update_coeffs);

    # test set params
    def test_set_params(self):
        ljgauss = md.pair.ljgauss(r_cut=3.0, nlist = self.nl);
        ljgauss.set_params(mode="no_shift");
        ljgauss.set_params(mode="shift");
        ljgauss.set_params(mode="xplor");
        self.assertRaises(RuntimeError, ljgauss.set_params, mode="blah");

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        ljgauss = md.pair.ljgauss(r_cut=2.5, nlist = self.nl);

        ljgauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=0.02**0.5, r0=1.1)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        ljgauss.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    def tearDown(self):
        del self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
