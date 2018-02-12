# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import cgcmm
from hoomd import md;
context.initialize()
import unittest
import os

# cgcmm.pair.cgcmm
class pair_cgcmm_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[4,4,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        cg = cgcmm.pair.cgcmm(r_cut=3.0, nlist = self.nl);
        cg.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='lj12_4');
        cg.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        cg = cgcmm.pair.cgcmm(r_cut=3.0, nlist = self.nl);
        cg.pair_coeff.set('A', 'A', sigma=1.0, alpha=1.0);
        self.assertRaises(RuntimeError, cg.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        cg = cgcmm.pair.cgcmm(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, cg.update_coeffs);

    # test nlist subscribe
    def test_nlist_subscribe(self):
        cg = cgcmm.pair.cgcmm(r_cut=2.5, nlist = self.nl);
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

    # test adding types
    def test_type_add(self):
        cg = cgcmm.pair.cgcmm(r_cut=3.0, nlist = self.nl);
        cg.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='lj12_4');
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, cg.update_coeffs);
        cg.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='lj12_4');
        cg.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='lj12_4');
        cg.update_coeffs();

    def tearDown(self):
        del self.s, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
