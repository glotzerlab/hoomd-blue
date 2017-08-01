# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.pair.table
class pair_table_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        table = md.pair.table(width=1000, nlist = self.nl);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        table.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        table = md.pair.table(width=1000, nlist = self.nl);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0);
        self.assertRaises(RuntimeError, table.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        table = md.pair.table(width=1000, nlist = self.nl);
        self.assertRaises(RuntimeError, table.update_coeffs);

    # test nlist subscribe
    def test_nlist_subscribe(self):
        table = md.pair.table(width=1000, nlist = self.nl);

        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        table.update_coeffs();
        self.nl.update_rcut();
        self.assertAlmostEqual(1.0, self.nl.r_cut.get_pair('A','A'));

        table.pair_coeff.set('A', 'A', rmax = 2.5)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

    # test adding types
    def test_type_add(self):
        table = md.pair.table(width=1000, nlist = self.nl);
        table.pair_coeff.set('A', 'A', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, table.update_coeffs);
        table.pair_coeff.set('A', 'B', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        table.pair_coeff.set('B', 'B', rmin=0.0, rmax=1.0, func=lambda r, rmin, rmax: (r, 2*r), coeff=dict());
        table.update_coeffs();

    def tearDown(self):
        del self.s, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
