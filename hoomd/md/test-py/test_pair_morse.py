# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.morse
class pair_morse_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        p = md.pair.morse(r_cut=3.0);
        p.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0, r_cut=3.0, r_on=2.0);
        p.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        p = md.pair.morse(r_cut=3.0);
        p.pair_coeff.set('A', 'A', D0=1.0);
        self.assertRaises(RuntimeError, p.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        p = md.pair.morse(r_cut=3.0);
        self.assertRaises(RuntimeError, p.update_coeffs);

    # test set params
    def test_set_params(self):
        p = md.pair.morse(r_cut=3.0);
        p.set_params(mode="no_shift");
        p.set_params(mode="shift");
        p.set_params(mode="xplor");
        self.assertRaises(RuntimeError, p.set_params, mode="blah");

    # test nlist global subscribe
    def test_nlist_global_subscribe(self):
        p = md.pair.morse(r_cut=2.5);
        p.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0)
        context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, context.current.neighbor_list.r_cut.get_pair('A','A'));

        p.pair_coeff.set('A', 'A', r_cut = 2.0)
        context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, context.current.neighbor_list.r_cut.get_pair('A','A'));

    # test nlist subscribe
    def test_nlist_subscribe(self):
        nl = md.nlist.cell()
        p = md.pair.morse(r_cut=2.5, nlist=nl);
        self.assertEqual(context.current.neighbor_list, None)

        p.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.5, nl.r_cut.get_pair('A','A'));

        p.pair_coeff.set('A', 'A', r_cut = 2.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.0, nl.r_cut.get_pair('A','A'));

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
