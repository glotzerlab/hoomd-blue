# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests force.constant
class concentration_modulation_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # test to see that se can create a force.concentration_modulation
    def test_create(self):
        mod = force.concentration_modulation();
        mod.coeff_set('A',A=1.0, i=3, w=0.02, p=3);
        all = group.all();
        integrate.mode_standard(dt=0.001);
        integrate.nve(all);
        run(100);


    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
