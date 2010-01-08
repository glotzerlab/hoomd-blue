# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for integrate.bdnvt
class integrate_bdnvt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        
    # tests basic creation of the integration method
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        bd = integrate.bdnvt(all, T=1.2, limit=0.1, seed=52);
        run(100);
        bd.disable();
        bd = integrate.bdnvt(all, T=1.2, limit=0.1);
        run(100);
        bd.disable();
        bd = integrate.bdnvt(all, T=1.2);
        run(100);
        bd.disable();
        bd = integrate.bdnvt(all, T=1.2, gamma_diam=True);
        bd.disable();
    
    # test set_params
    def test_set_params(self):
        all = group.all();
        bd = integrate.bdnvt(all, T=1.2);
        bd.set_params(T=1.3);

    # test set_gamma
    def test_set_gamma(self):
        all = group.all();
        bd = integrate.bdnvt(all, T=1.2);
        bd.set_gamma('A', 0.5);
        bd.set_gamma('B', 1.0);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
