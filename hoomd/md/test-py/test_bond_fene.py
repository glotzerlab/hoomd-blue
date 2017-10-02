# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os

# tests md.bond.fene
class bond_fene_tests (unittest.TestCase):
    def setUp(self):
        print
        init.read_gsd(os.path.join(os.path.dirname(__file__),'test_data_polymer_system.gsd'));
        context.current.sorter.set_params(grid=8)

    # test to see that se can create a md.force.constant
    def test_create(self):
        md.bond.fene();

    # test setting coefficients
    def test_set_coeff(self):
        fene = md.bond.fene();
        fene.bond_coeff.set('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon=2.0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test integrating with a zero force constant
    def test_zero_coeff(self):
        fene = md.bond.fene();
        fene.bond_coeff.set('polymer', k=0.0, r0=.001, sigma=1.0, epsilon=0)
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

    # test coefficient not set checking
    def test_set_coeff_fail(self):
        fene = md.bond.fene();
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
