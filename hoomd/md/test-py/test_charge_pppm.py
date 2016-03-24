# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

# md.charge.pppm
class charge_pppm_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)
        for i in range(0,50):
            self.s.particles[i].charge = -1;

        for i in range(50,100):
            self.s.particles[i].charge = 1;

    # basic test of creation and param setting
    def test(self):
        all = group.all()
        c = md.charge.pppm(all);
        c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

        del all
        del c


    # Cannot test pppm multiple times currently because of implementation limitations
    ## test missing coefficients
    #def test_set_missing_coeff(self):
        #all = group.all()
        #c = md.charge.pppm(all);
        #integrate.mode_standard(dt=0.005);
        #integrate.nve(all);
        #self.assertRaises(RuntimeError, run, 100);

    ## test enable/disable
    #def test_enable_disable(self):
        #all = group.all()
        #c = md.charge.pppm(all);
        #c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);

        #c.disable(log=True);
        #c.enable();

    def tearDown(self):
        del self.s
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
