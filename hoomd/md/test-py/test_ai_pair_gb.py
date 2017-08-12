# -*- coding: iso-8859-1 -*-
# Maintainer: mspells

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

# md.pair.gb
class pair_gb_tests (unittest.TestCase):
    def setUp(self):
        print
        self.system = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]);
        snap = self.system.take_snapshot(all=True)
        snap.particles.angmom[:] = 1
        self.system.restore_snapshot(snap)

        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # basic test of creation
    def test(self):
        gb = md.pair.gb(r_cut=3.0, nlist = self.nl);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0, lpar=1.5);
        gb.update_coeffs();

    # test missing coefficients
    def test_set_missing_epsilon(self):
        gb = md.pair.gb(r_cut=3.0, nlist = self.nl);
        gb.pair_coeff.set('A', 'A', lperp=1.0, lpar=1.5);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_set_missing_lperp(self):
        gb = md.pair.gb(r_cut=3.0, nlist = self.nl);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lpar=1.5);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_set_missing_lpar(self):
        gb = md.pair.gb(r_cut=3.0, nlist = self.nl);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test missing coefficients
    def test_missing_AA(self):
        gb = md.pair.gb(r_cut=3.0, nlist = self.nl);
        self.assertRaises(RuntimeError, gb.update_coeffs);

    # test set params
    def test_set_params(self):
        gb = md.pair.gb(r_cut=3.0, nlist = self.nl);
        gb.set_params(mode="no_shift");
        gb.set_params(mode="shift");
        # xplor is not implemented for anisotropic pair potentials
        self.assertRaises(RuntimeError, gb.set_params, mode="blah");

    # test nlist subscribe
    def test_nlist_subscribe(self):
        gb = md.pair.gb(r_cut=2.5, nlist = self.nl);
        gb.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','A'));

        gb.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.nl.update_rcut();
        self.assertAlmostEqual(2.0, self.nl.r_cut.get_pair('A','A'));

    # test adding types
    def test_type_add(self):
        gb = md.pair.gb(r_cut=2.5, nlist = self.nl);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0, lpar=1.5);
        self.system.particles.types.add('B')
        self.assertRaises(RuntimeError, gb.update_coeffs);
        gb.pair_coeff.set(['A','B'], 'B', epsilon=1.0, lperp=1.0, lpar=1.5);
        gb.update_coeffs();

    def tearDown(self):
        del self.system,self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
