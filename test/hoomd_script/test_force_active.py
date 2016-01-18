# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os, math, numpy as np

# tests force.active
class force_active_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # test to see that can create a force.active
    def test_create(self):
        np.random.seed(MySeed)
        activity = [ tuple(((np.random.rand(3) - 0.5) * 2.0)) for i in range(100)] # random forces
        force.active(seed=7, f_lst=activity)
        
    # tests options to force.active
    def test_options(self):
        np.random.seed(MySeed)
        activity = [ tuple(((np.random.rand(3) - 0.5) * 2.0)) for i in range(100)] # random forces
        force.active(seed=2, f_lst=activity, rotation_diff=1.0)
        force.active(seed=2, f_lst=activity, orientation_link=False)
        ellipsoid = update.constraint_ellipsoid(P=(0,0,0), rx=3, ry=3, rz=3)
        force.active(seed=2, f_lst=activity, constraint=ellipsoid)

    # test the initialization checks
    def test_init_checks(self):
        act = force.active();
        act.cpp_force = None;

        self.assertRaises(RuntimeError, const.enable);
        self.assertRaises(RuntimeError, const.disable);
        self.assertRaises(RuntimeError, const.benchmark, 500);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
