# -*- coding: iso-8859-1 -*-
# Maintainer: jamesaan

from hoomd_script import *
import unittest
import os

# pair.mie
class pair_mie_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        mie = pair.mie(r_cut=3.0);
        mie.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, n=13.0, m=7.0, r_cut=2.5, r_on=2.0);
        mie.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        mie = pair.mie(r_cut=3.0);
        mie.pair_coeff.set('A', 'A', sigma=1.0, n=13.0, m=7.0);
        self.assertRaises(RuntimeError, mie.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        mie = pair.mie(r_cut=3.0);
        self.assertRaises(RuntimeError, mie.update_coeffs);

    # test set params
    def test_set_params(self):
        mie = pair.mie(r_cut=3.0);
        mie.set_params(mode="no_shift");
        mie.set_params(mode="shift");
        mie.set_params(mode="xplor");
        self.assertRaises(RuntimeError, mie.set_params, mode="blah");

    # test default coefficients
    def test_default_coeff(self):
        mie = pair.mie(r_cut=3.0);
        # (alpha, r_cut, and r_on are default)
        mie.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, n=13.0, m=7.0)
        mie.update_coeffs()

    # test max rcut
    def test_max_rcut(self):
        mie = pair.mie(r_cut=2.5);
        mie.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0, n=13.0, m=7.0)
        self.assertAlmostEqual(2.5, mie.get_max_rcut());
        mie.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.0, mie.get_max_rcut());

    # test nlist subscribe
    def test_nlist_subscribe(self):
        mie = pair.mie(r_cut=2.5);
        mie.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0, n=13.0, m=7.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut);

        mie.pair_coeff.set('A', 'A', r_cut = 2.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, globals.neighbor_list.r_cut);

    # test coeff list
    def test_coeff_list(self):
        mie = pair.mie(r_cut=3.0);
        mie.pair_coeff.set(['A', 'B'], ['A', 'C'], epsilon=1.0, sigma=1.0, n=13.0, m=7.0, r_cut=2.5, r_on=2.0);
        mie.update_coeffs();

    # test adding types
    def test_type_add(self):
        mie = pair.mie(r_cut=3.0);
        mie.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, n=13.0, m=7.0);
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, mie.update_coeffs);
        mie.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, n=13.0, m=7.0)
        mie.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, n=13.0, m=7.0)
        mie.update_coeffs();

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
