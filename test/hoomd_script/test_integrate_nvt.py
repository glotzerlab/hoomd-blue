# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for integrate.nvt
class integrate_nvt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        import __main__;
        __main__.sorter.set_params(grid=8)
        
    # tests basic creation of the dump
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nvt(all, T=1.2, tau=0.5);
        run(100);
    
    # test set_params
    def test_set_params(self):
        all = group.all();
        nvt = integrate.nvt(all, T=1.2, tau=0.5);
        nvt.set_params(T=1.3);
        nvt.set_params(tau=0.6);
    
    # test w/ empty group
    def test_empty(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = integrate.mode_standard(dt=0.005);
        nvt = integrate.nvt(group=empty, T=1.0, tau=0.5)
        run(1);
    
    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

