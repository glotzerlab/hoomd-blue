# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# pair - multiple type max_rcut test
class pair_max_rcut_tests (unittest.TestCase):
    def setUp(self):
        #print
        create_empty(N=100, box=data.boxdim(L=40), particle_types=['A', 'B']);
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    def test_max_rcut(self):
        lj = md.pair.lj(r_cut=2.5, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0)
        lj.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0)
        lj.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, lj.get_max_rcut());
        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.5, lj.get_max_rcut());
        lj.pair_coeff.set('A', 'B', r_cut = 3.0)
        self.assertAlmostEqual(3.0, lj.get_max_rcut());
        lj.pair_coeff.set('B', 'B', r_cut = 3.5)
        self.assertAlmostEqual(3.5, lj.get_max_rcut());

    def test_nlist_subscribe(self):
        lj = md.pair.lj(r_cut=2.5, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=3.0)
        lj.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0)
        lj.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=3.1)

        # check that everything is initialized correctly
        self.nl.update_rcut()
        self.assertAlmostEqual(3.0, self.nl.r_cut.get_pair('A','A'));
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','B'));
        self.assertAlmostEqual(3.1, self.nl.r_cut.get_pair('B','B'));

        # update a pair coefficient, and check
        lj.pair_coeff.set('A','B', r_cut = 5.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(5.0, self.nl.r_cut.get_pair('B','A'))

        # a second potential, only (B,B) should be bigger than the LJ
        gauss = md.pair.gauss(r_cut=1.0, nlist = self.nl)
        gauss.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=1.0)
        gauss.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0, r_cut=2.0)
        gauss.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=5.1)

        self.nl.update_rcut()
        self.assertAlmostEqual(3.0, self.nl.r_cut.get_pair('A','A'));
        self.assertAlmostEqual(5.0, self.nl.r_cut.get_pair('B','A'));
        self.assertAlmostEqual(5.1, self.nl.r_cut.get_pair('B','B'));

        # change B,B back down, and make sure you get the LJ cutoff instead
        gauss.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=1.0)
        run(1)
        self.assertAlmostEqual(3.1, self.nl.r_cut.get_pair('B','B'));

    # test independent subscription to two neighbor list
    def test_multi_nlist_subscribe(self):
        lj = md.pair.lj(r_cut=2.5, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=3.0)
        lj.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0)
        lj.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=3.1)

        # check that everything is initialized correctly
        self.nl.update_rcut()
        self.assertAlmostEqual(3.0, self.nl.r_cut.get_pair('A','A'));
        self.assertAlmostEqual(2.5, self.nl.r_cut.get_pair('A','B'));
        self.assertAlmostEqual(3.1, self.nl.r_cut.get_pair('B','B'));

        # update a pair coefficient, and check
        lj.pair_coeff.set('A','B', r_cut = 5.0)
        self.nl.update_rcut()
        self.assertAlmostEqual(5.0, self.nl.r_cut.get_pair('B','A'))

        # a second potential, only (B,B) should be bigger than the LJ
        # but, it's attached to the second neighbor list, so nothing should change
        nl2 = md.nlist.cell()
        gauss = md.pair.gauss(r_cut=1.0, nlist = nl2)
        gauss.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=1.0)
        gauss.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0, r_cut=2.0)
        gauss.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=5.1)

        self.nl.update_rcut()
        nl2.update_rcut()
        self.assertAlmostEqual(3.0, self.nl.r_cut.get_pair('A','A'));
        self.assertAlmostEqual(5.0, self.nl.r_cut.get_pair('A','B'));
        self.assertAlmostEqual(3.1, self.nl.r_cut.get_pair('B','B'));
        self.assertAlmostEqual(1.0, nl2.r_cut.get_pair('A','A'));
        self.assertAlmostEqual(2.0, nl2.r_cut.get_pair('B','A'));
        self.assertAlmostEqual(5.1, nl2.r_cut.get_pair('B','B'));

        # now, attach a third potential to the first neighbor list, and things should change there
        slj = md.pair.slj(r_cut=1.0, nlist=self.nl, d_max=1.0)
        slj.pair_coeff.set('A', 'A', sigma=1.0, epsilon=1.0, r_cut=1.0)
        slj.pair_coeff.set('A', 'B', sigma=1.0, epsilon=1.0, r_cut=2.0)
        slj.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=5.1)

        self.nl.update_rcut()
        nl2.update_rcut()
        self.assertAlmostEqual(3.0, self.nl.r_cut.get_pair('A','A'));
        self.assertAlmostEqual(5.0, self.nl.r_cut.get_pair('B','A'));
        self.assertAlmostEqual(5.1, self.nl.r_cut.get_pair('B','B'));

        # change B,B back down, and make sure you get the LJ cutoff instead
        slj.pair_coeff.set('B', 'B', sigma=1.0, epsilon=1.0, r_cut=1.0)
        run(1)
        self.assertAlmostEqual(3.1, self.nl.r_cut.get_pair('B','B'));

    def tearDown(self):
        del self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
