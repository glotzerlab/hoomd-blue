# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair.table
class pair_table_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        
    # basic test of creation
    def test(self):
        table = pair.table(width=1000);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        table.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        table = pair.table(width=1000);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0);
        self.assertRaises(RuntimeError, table.update_coeffs);
        
    # test missing coefficients
    def test_missing_AA(self):
        table = pair.table(width=1000);
        self.assertRaises(RuntimeError, table.update_coeffs);

    # test nlist subscribe
    def teat_nlist_subscribe(self):
        table = pair.table(width=1000);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        table.update_coeffs();
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(1.0, globals.neighbor_list.r_cut);
        
        table.pair_coeff.set('A', 'A', rmax = 2.5)
        globals.neighbor_list.update_rcut();
        self.assertAlmostEqual(2.5, globals.neighbor_list.r_cut);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

