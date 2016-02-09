# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# pair.gauss
class pair_gauss_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);
        gauss.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.pair_coeff.set('A', 'A', sigma=1.0);
        self.assertRaises(RuntimeError, gauss.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        gauss = pair.gauss(r_cut=3.0);
        self.assertRaises(RuntimeError, gauss.update_coeffs);

    # test set params
    def test_set_params(self):
        gauss = pair.gauss(r_cut=3.0);
        gauss.set_params(mode="no_shift");
        gauss.set_params(mode="shift");
        gauss.set_params(mode="xplor");
        self.assertRaises(RuntimeError, gauss.set_params, mode="blah");

    # test global nlist subscribe
    def test_nlist_global_subscribe(self):
        gauss = pair.gauss(r_cut=2.5);
        gauss.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        hoomd_script.context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, hoomd_script.context.current.neighbor_list.r_cut.get_pair('A','A'));

        gauss.pair_coeff.set('A', 'A', r_cut = 2.0)
        hoomd_script.context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, hoomd_script.context.current.neighbor_list.r_cut.get_pair('A','A'));

    # test specific nlist subscription
    def test_nlist_subscribe(self):
        nl = nlist.cell()
        gauss = pair.gauss(r_cut=2.5, nlist = nl);
        self.assertEqual(hoomd_script.context.current.neighbor_list, None)

        gauss.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.5, nl.r_cut.get_pair('A','A'));

        gauss.pair_coeff.set('A', 'A', r_cut = 2.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.0, nl.r_cut.get_pair('A','A'));

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
