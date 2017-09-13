# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.lj
class pair_lj_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0);
        lj.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', sigma=1.0, alpha=1.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test set params
    def test_set_params(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.set_params(mode="no_shift");
        lj.set_params(mode="shift");
        lj.set_params(mode="xplor");
        self.assertRaises(RuntimeError, lj.set_params, mode="blah");

    # test default coefficients
    def test_default_coeff(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        # (alpha, r_cut, and r_on are default)
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        lj.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        lj = md.pair.lj(r_cut=2.5, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, lj.get_max_rcut());
        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, lj.get_max_rcut());

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        lj = md.pair.lj(r_cut=2.5, nlist = self.nl);

        lj.pair_coeff.set('A', 'A', sigma = 1.0, epsilon=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    # test coeff list
    def test_coeff_list(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, alpha=1.0, r_cut=2.5, r_on=2.0);
        lj.update_coeffs();

    # test adding types
    def test_type_add(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, lj.update_coeffs);
        lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0)
        lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0)
        lj.update_coeffs();

    # test that pair coefficients can be added and set using unicode strings
    def test_unicode_type(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set(u'A', u'A', epsilon=1.0, sigma=1.0);
        self.s.particles.types.add(u'Bb')
        self.assertRaises(RuntimeError, lj.update_coeffs);
        lj.pair_coeff.set(u'A', u'Bb', epsilon=1.0, sigma=1.0)
        lj.pair_coeff.set(u'Bb', u'Bb', epsilon=1.0, sigma=1.0)
        lj.update_coeffs();

    def tearDown(self):
        del self.s, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
