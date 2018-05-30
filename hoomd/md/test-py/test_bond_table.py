# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# tests md.bond.bondtable
class bond_table_tests (unittest.TestCase):
    def setUp(self):
        print
        init.read_gsd(os.path.join(os.path.dirname(__file__),'test_data_polymer_system.gsd'));
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        btable = md.bond.table(width=1000);
        btable.bond_coeff.set('polymer', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        btable.update_coeffs();

    # test missing coefficients
    def test_set_missing_coeff(self):
        btable = md.bond.table(width=1000);
        btable.bond_coeff.set('polymer', rmin=0.0, rmax=1.0);
        self.assertRaises(RuntimeError, btable.update_coeffs);

    # test more missing coefficients
    def test_missing_all(self):
        btable = md.bond.table(width=1000);
        self.assertRaises(RuntimeError, btable.update_coeffs);


    # Add tests to check for runtime errors

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
