# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests bond.bondtable
class bond_table_tests (unittest.TestCase):
    def setUp(self):
        print
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = data.boxdim(L=35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);

        sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        btable = bond.table(width=1000);
        btable.bond_coeff.set('polymer', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        btable.update_coeffs();

    # test missing coefficients
    def test_set_missing_coeff(self):
        btable = bond.table(width=1000);
        btable.bond_coeff.set('polymer', rmin=0.0, rmax=1.0);
        self.assertRaises(RuntimeError, btable.update_coeffs);

    # test more missing coefficients
    def test_missing_all(self):
        btable = bond.table(width=1000);
        self.assertRaises(RuntimeError, btable.update_coeffs);


    # Add tests to check for runtime errors

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
