# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# pair - multiple type max_rcut test
class pair_max_rcut_tests (unittest.TestCase):
    def setUp(self):
        #print
        init.create_empty(N=100, box=(20,20,20), n_particle_types=2);
        import __main__;
        __main__.sorter.set_params(grid=8)

    def test_max_rcut(self):
        lj = pair.lj(r_cut=2.5);
        lj.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0)
        lj.pair_coeff.set('A', 'B', simga=1.0, epsilon=1.0)
        lj.pair_coeff.set('B', 'B', simga=1.0, epsilon=1.0)
        self.assertAlmostEqual(2.5, lj.get_max_rcut());
        lj.pair_coeff.set('A', 'A', r_cut = 2.0)
        self.assertAlmostEqual(2.5, lj.get_max_rcut());
        lj.pair_coeff.set('A', 'B', r_cut = 3.0)
        self.assertAlmostEqual(3.0, lj.get_max_rcut());
        lj.pair_coeff.set('B', 'B', r_cut = 3.5)
        self.assertAlmostEqual(3.5, lj.get_max_rcut());

    def test_nlist_subscribe(self):
        lj = pair.lj(r_cut=2.5);
        lj.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0, r_cut=3.0)
        lj.pair_coeff.set('A', 'B', simga=1.0, epsilon=1.0, r_cut=2.5)
        lj.pair_coeff.set('B', 'B', simga=1.0, epsilon=1.0, r_cut=3.1)

        globals.neighbor_list.update_rcut()
        self.assertAlmostEqual(3.1, globals.neighbor_list.r_cut);

        gauss = pair.gauss(r_cut=1.0)
        gauss.pair_coeff.set('A', 'A', simga=1.0, epsilon=1.0, r_cut=1.0)
        gauss.pair_coeff.set('A', 'B', simga=1.0, epsilon=1.0, r_cut=2.0)
        gauss.pair_coeff.set('B', 'B', simga=1.0, epsilon=1.0, r_cut=5.1)

        globals.neighbor_list.update_rcut()
        self.assertAlmostEqual(5.1, globals.neighbor_list.r_cut);

        gauss.pair_coeff.set('B', 'B', simga=1.0, epsilon=1.0, r_cut=1.0)
        run(1)
        self.assertAlmostEqual(3.1, globals.neighbor_list.r_cut);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
