# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

# tests md.force.constant
class external_periodic_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

    # test to see that se can create an external.periodic
    def test_create(self):
        periodic = md.external.periodic();
        periodic.force_coeff.set('A',A=1.0, i=3, w=0.02, p=3);
        all = group.all();
        md.integrate.mode_standard(dt=0.001);
        md.integrate.nve(all);
        run(100);

    ## test enable/disable
    def test_enable_disable(self):
        all = group.all()
        periodic = md.external.periodic();
        periodic.force_coeff.set('A',A=1.0, i=3, w=0.02, p=3);

        periodic.disable();
        periodic.enable();

    # test adding types
    def test_type_add(self):
        periodic = md.external.periodic();
        periodic.force_coeff.set('A',A=1.0, i=3, w=0.02, p=3);
        self.s.particles.types.add('B')
        self.assertRaises(RuntimeError, periodic.update_coeffs);
        periodic.force_coeff.set('A',A=1.0, i=3, w=0.02, p=3);
        periodic.force_coeff.set('B',A=1.0, i=3, w=0.02, p=3);
        periodic.update_coeffs();

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
