# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.slj
class pair_slj_tests (unittest.TestCase):
    def setUp(self):
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = md.pair.slj(r_cut=3.0, nlist = self.nl, d_max=2.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, r_cut=2.5);
        lj.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj = md.pair.slj(r_cut=3.0, nlist = self.nl, d_max=2.0);
        lj.pair_coeff.set('A', 'A', sigma=1.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        lj = md.pair.slj(r_cut=3.0, nlist = self.nl, d_max=2.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test set params
    def test_set_params(self):
        lj = md.pair.slj(r_cut=3.0, nlist = self.nl, d_max=2.0);
        lj.set_params(mode="no_shift");
        lj.set_params(mode="shift");
        self.assertRaises(RuntimeError, lj.set_params, mode="xplor");
        self.assertRaises(RuntimeError, lj.set_params, mode="blah");

    # test default coefficients
    def test_default_coeff(self):
        lj = md.pair.slj(r_cut=3.0, nlist = self.nl, d_max=2.0);
        # (r_cut, and r_on are default)
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        lj.update_coeffs()

    # test nlist subscribe
    def test_nlist_subscribe(self):
        lj = md.pair.slj(r_cut=2.5, nlist = self.nl, d_max=2.0);

        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    # test set params
    def test_dmax(self):
        import random
        random.seed(123)
        rmax = -10.0
        for p in self.s.particles:
            r = random.uniform(-1,1)
            if r > rmax:
                rmax = r
            p.diameter = r

        lj = md.pair.slj(r_cut=3.0, nlist = self.nl);
        self.assertAlmostEqual(self.nl.cpp_nlist.getMaximumDiameter(), rmax,5)

    def tearDown(self):
        del self.s, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
