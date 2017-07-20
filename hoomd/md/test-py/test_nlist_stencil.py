# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# md.nlist.stencil testing
class nlist_stencil_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[10,10,10]); #target a packing fraction of 0.05

        # directly create a neighbor list
        self.nl = md.nlist.stencil()

        context.current.sorter.set_params(grid=8)

    # test set_params
    def test_set_params(self):
        self.nl.set_params(r_buff=0.6);
        self.nl.set_params(check_period = 20);
        self.nl.set_params(d_max = 2.0, dist_check = False)

    # test reset_exclusions
    def test_reset_exclusions_works(self):
        self.nl.reset_exclusions();
        self.nl.reset_exclusions(exclusions = ['1-2']);
        self.nl.reset_exclusions(exclusions = ['1-3']);
        self.nl.reset_exclusions(exclusions = ['1-4']);
        self.nl.reset_exclusions(exclusions = ['bond']);
        self.nl.reset_exclusions(exclusions = ['angle']);
        self.nl.reset_exclusions(exclusions = ['dihedral']);
        self.nl.reset_exclusions(exclusions = ['pair']);
        self.nl.reset_exclusions(exclusions = ['bond', 'angle']);

    # test reset_exclusions error messages
    def test_reset_exclusions_nowork(self):
        self.assertRaises(RuntimeError,
                          self.nl.reset_exclusions,
                          exclusions = ['bond', 'angle', 'invalid']);

    # test tuning
    def test_tune(self):
        self.nl.tune(warmup=100, r_min=0.1, r_max=0.25, jumps=10, steps=50)

    # test cell width tuning
    def test_tune_cell_width(self):
        self.nl.tune_cell_width(warmup=100, jumps=10, steps=50)
        self.nl.tune_cell_width(warmup=10, min_width=0.1, max_width=0.25, jumps=5, steps=5)

    # test multiple neighbor lists can coexist with different parameters
    def test_multi(self):
        self.nl.set_params(r_buff = 0.3)

        nl2 = md.nlist.stencil()
        nl2.set_params(r_buff = 0.8)

        self.assertEqual(self.nl.r_buff, 0.3)
        self.assertEqual(nl2.r_buff, 0.8)

        lj1 = md.pair.lj(r_cut = 2.0, nlist = self.nl)
        lj2 = md.pair.lj(r_cut = 3.0, nlist = nl2)
        lj3 = md.pair.lj(r_cut = 4.0, nlist = nl2)

        # check that each neighbor list has the right cutoff
        self.assertAlmostEqual(self.nl.r_cut.get_pair('A','A'), 2.0)
        self.assertAlmostEqual(nl2.r_cut.get_pair('A','A'), 4.0)

        # force an update to trigger and recheck that the right coefficients updated
        lj1.pair_coeff.set('A','A', r_cut = 5.0)
        run(1)
        self.assertAlmostEqual(self.nl.r_cut.get_pair('A','A'), 5.0)
        self.assertAlmostEqual(nl2.r_cut.get_pair('A','A'), 4.0)

    def tearDown(self):
        del self.nl
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
