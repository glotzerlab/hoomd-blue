# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.yukawa
class pair_yukawa_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        yuk = md.pair.yukawa(r_cut=3.0);
        yuk.pair_coeff.set('A', 'A', epsilon=1.0, kappa=1.0);
        yuk.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        yuk = md.pair.yukawa(r_cut=3.0);
        yuk.pair_coeff.set('A', 'A', kappa=1.0);
        self.assertRaises(RuntimeError, yuk.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        yuk = md.pair.yukawa(r_cut=3.0);
        self.assertRaises(RuntimeError, yuk.update_coeffs);

    # test set params
    def test_set_params(self):
        yuk = md.pair.yukawa(r_cut=3.0);
        yuk.set_params(mode="no_shift");
        yuk.set_params(mode="shift");
        yuk.set_params(mode="xplor");
        self.assertRaises(RuntimeError, yuk.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_global_subscribe(self):
        yuk = md.pair.yukawa(r_cut=2.5);
        yuk.pair_coeff.set('A', 'A', epsilon=1.0, kappa=1.0)
        context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, context.current.neighbor_list.r_cut.get_pair('A','A'));

        yuk.pair_coeff.set('A', 'A', r_cut = 2.0)
        context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, context.current.neighbor_list.r_cut.get_pair('A','A'));

    # test nlist subscribe
    def test_nlist_subscribe(self):
        nl = md.nlist.cell()
        yuk = md.pair.yukawa(r_cut=2.5, nlist=nl);
        self.assertEqual(context.current.neighbor_list, None)

        yuk.pair_coeff.set('A', 'A', epsilon=1.0, kappa=1.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.5, nl.r_cut.get_pair('A','A'));

        yuk.pair_coeff.set('A', 'A', r_cut = 2.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.0, nl.r_cut.get_pair('A','A'));

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
