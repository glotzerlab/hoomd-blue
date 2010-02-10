# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for integrate.nve
class integrate_nve_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
                
    # tests basic creation of the dump
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
    
    # tests creation of the method with options
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all, limit=0.01, zero_force=True);
        run(100);
    
    # test set_params
    def test_set_params(self):
        all = group.all();
        mode = integrate.mode_standard(dt=0.005);
        mode.set_params(dt=0.001);
        nve = integrate.nve(all);
        nve.set_params(limit=False);
        nve.set_params(limit=0.1);
        nve.set_params(zero_force=False);

    # test w/ empty group
    def test_empty(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = integrate.mode_standard(dt=0.005);
        nve = integrate.nve(group=empty)
        run(1);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

