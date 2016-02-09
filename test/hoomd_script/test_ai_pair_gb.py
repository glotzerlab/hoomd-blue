# -*- coding: iso-8859-1 -*-
# Maintainer: mspells

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# pair.gb
class pair_gb_tests (unittest.TestCase):
    def setUp(self):
        print
        system = init.create_random(N=100, phi_p=0.05);
        snap = system.take_snapshot(all=True)
        snap.particles.angmom[:] = 1
        system.restore_snapshot(snap)

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        gb = pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0, lpar=1.5);
        gb.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        gb = pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', lperp=1.0, lpar=1.5);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_set_missing_lperp(self):
        gb = pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lpar=1.5);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_set_missing_lpar(self):
        gb = pair.gb(r_cut=3.0);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        gb = pair.gb(r_cut=3.0);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test set params
    def test_set_params(self):
        gb = pair.gb(r_cut=3.0);
        gb.set_params(mode="no_shift");
        gb.set_params(mode="shift");
        # xplor is not implemented for anisotropic pair potentials
        self.assertRaises(RuntimeError, gb.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_subscribe(self):
        gb = pair.gb(r_cut=2.5);
        gb.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        hoomd_script.context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, hoomd_script.context.current.neighbor_list.r_cut.get_pair('A','A'));

        gb.pair_coeff.set('A', 'A', r_cut = 2.0)
        hoomd_script.context.current.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, hoomd_script.context.current.neighbor_list.r_cut.get_pair('A','A'));

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
