# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for integrate.npt
class integrate_npt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        
    # tests basic creation of the dump
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5);
        run(100);
    
    # test set_params
    def test_set_params(self):
        integrate.mode_standard(dt=0.005);
        all = group.all();
        npt = integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5);
        npt.set_params(T=1.3);
        npt.set_params(tau=0.6);
        npt.set_params(P=0.5);
        npt.set_params(tauP=0.6);
        npt.set_params(partial_scale=True);
        run(100);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

