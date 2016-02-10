# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# pair.slj
class pair_slj_tests (unittest.TestCase):
    def setUp(self):
        self.s = init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, r_cut=2.5);
        lj.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        lj.pair_coeff.set('A', 'A', sigma=1.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        self.assertRaises(RuntimeError, lj.update_coeffs);

    # test set params
    def test_set_params(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        lj.set_params(mode="no_shift");
        lj.set_params(mode="shift");
        self.assertRaises(RuntimeError, lj.set_params, mode="xplor");
        self.assertRaises(RuntimeError, lj.set_params, mode="blah");

    # test default coefficients
    def test_default_coeff(self):
        lj = pair.slj(r_cut=3.0, d_max=2.0);
        # (r_cut, and r_on are default)
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        lj.update_coeffs()

    # test nlist global subscribe
    def test_nlist_subscribe(self):
        lj = pair.slj(r_cut=2.5, d_max=2.0);
        lj.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        hoomd_script.context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, hoomd_script.context.current.neighbor_list.r_cut.get_pair('A','A'));

        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        hoomd_script.context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, hoomd_script.context.current.neighbor_list.r_cut.get_pair('A','A'));

    # test nlist subscribe
    def test_nlist_subscribe(self):
        nl = nlist.cell()
        lj = pair.slj(r_cut=2.5, d_max=2.0, nlist=nl);
        self.assertEqual(hoomd_script.context.current.neighbor_list, None)

        lj.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.5, nl.r_cut.get_pair('A','A'));

        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        nl.update_rcut();
        self.assertAlmostEqual(2.0, nl.r_cut.get_pair('A','A'));

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

        lj = pair.slj(r_cut=3.0);
        self.assertAlmostEqual(hoomd_script.context.current.neighbor_list.cpp_nlist.getMaximumDiameter(), rmax,5)

    def tearDown(self):
        del self.s
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
