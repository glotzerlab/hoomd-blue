# -*- coding: iso-8859-1 -*-
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
        import __main__;
        __main__.sorter.set_params(grid=8)
        
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
        run(100);
        bd.disable();
        bd = integrate.bdnvt(all, T=1.2, gamma_diam=True,tally=True);
        run(100);
        bd.disable();
    
    # test set_params
    def test_set_params(self):
        all = group.all();
        bd = integrate.bdnvt(all, T=1.2);
        bd.set_params(T=1.3);
        bd.set_params(tally=False);

    # test set_gamma
    def test_set_gamma(self):
        all = group.all();
        bd = integrate.bdnvt(all, T=1.2);
        bd.set_gamma('A', 0.5);
        bd.set_gamma('B', 1.0);

#FIXME
#    # test w/ empty group
#    def test_empty(self):
#        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
#        mode = integrate.mode_standard(dt=0.005);
#        nve = integrate.bdnvt(group=empty, T=1.2)
#        run(1);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

