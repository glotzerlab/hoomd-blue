# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os

# tests for update.box_resize
class update_box_resize_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        update.box_resize(L = variant.linear_interp([(0, 20), (1e6, 50)]))
        run(100);

    # tests with phase
    def test_phase(self):
        update.box_resize(L = variant.linear_interp([(0, 20), (1e6, 50)]), period=10, phase=0)
        run(100);

    # test the setting of more args
    def test_moreargs(self):
        update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]),
                        Ly = variant.linear_interp([(0, 40), (1e6, 80)]))
        run(100);

    # test the setting of more args
    def test_evenmoreargs(self):
        update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]),
                        Ly = variant.linear_interp([(0, 40), (1e6, 80)]),
                        Lz = variant.linear_interp([(0, 40), (1e6, 80)]),
                        period=10);
        run(100);

    # shear test
    def test_shear(self):
        update.box_resize(xy= variant.linear_interp([(0,0), (1e5, 1)]), period=10);
        run(100);

    # shear test with more shear planes
    def test_shear_more_planes(self):
        update.box_resize(xy= variant.linear_interp([(0,0), (1e5, 1)]),
                          xz= variant.linear_interp([(0,0), (1e5, .5)]),
                          yz= variant.linear_interp([(0,0), (1e5, .3)]), period=10);
        run(100);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
