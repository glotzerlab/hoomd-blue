# -*- coding: iso-8859-1 -*-
# Maintainer: mspells

from hoomd_script import *
import unittest
import os

# ai_pair.dipole
class ai_pair_dipole_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        dipole = ai_pair.dipole(r_cut=3.0);
        dipole.pair_coeff.set('A', 'A', mu=1.0, A=1.0, kappa=1.0)
        dipole.update_coeffs();

    # test missing coefficients
    def test_set_missing_mu(self):
        dipole = ai_pair.dipole(r_cut=3.0);
        dipole.pair_coeff.set('A', 'A', A=1.0, kappa=1.0)
        self.assertRaises(RuntimeError, dipole.update_coeffs);

    # test missing coefficients
    def test_set_missing_kappa(self):
        dipole = ai_pair.dipole(r_cut=3.0);
        dipole.pair_coeff.set('A', 'A', mu=1.0, A=1.0)
        self.assertRaises(RuntimeError, dipole.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        dipole = ai_pair.dipole(r_cut=3.0);
        self.assertRaises(RuntimeError, dipole.update_coeffs);

    # test set params
    def test_set_params(self):
        dipole = ai_pair.dipole(r_cut=3.0);
        dipole.set_params(mode="no_shift");
        dipole.set_params(mode="shift");
        # xplor is not implemented for anisotropic pair potentials
        self.assertRaises(RuntimeError, dipole.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_subscribe(self):
        dipole = ai_pair.dipole(r_cut=2.5);
        dipole.pair_coeff.set('A', 'A', mu=1.0, kappa=1.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut.get_pair('A','A'));

        dipole.pair_coeff.set('A', 'A', r_cut = 2.0)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.0, globals.neighbor_list.r_cut.get_pair('A','A'));

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
