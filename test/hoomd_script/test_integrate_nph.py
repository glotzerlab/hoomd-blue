# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for integrate.nph
class integrate_nph_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        import __main__;
        __main__.sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.001);
        integrate.nph(all, P=1.0, W=.001, mode = "cubic");
        run(100);

    # test set_params
    def test_set_params(self):
        integrate.mode_standard(dt=0.001);
        all = group.all();
        nph = integrate.nph(all, P=1.0, W=.001);
        nph.set_params(P=0.5);
        nph.set_params(W=0.6);
        run(100);
        nph.set_params(mode="orthorhombic")
        run(100);
        nph.set_params(mode="tetragonal")
        run(100);

    # test w/ empty group
    def test_empty(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = integrate.mode_standard(dt=0.005);
        nph = integrate.nph(group=empty, P=1.0, W=1)
        run(1);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
