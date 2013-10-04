# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests wall.lj
class wall_lj_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # test to see that se can create a wall.lj
    def test_create(self):
        wall.lj(r_cut=3.0);

    # test setting coefficients
    def test_set_coeff(self):
        lj_wall = wall.lj(r_cut=3.0);
        lj_wall.set_coeff('A', epsilon=1.0, sigma=1.0, alpha=1.0)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        lj_wall = wall.lj(r_cut=3.0);
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
