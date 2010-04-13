# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests angle.harmonic
class angle_harmonic_tests (unittest.TestCase):
    def setUp(self):
        print
        # create a polymer system and add a few angles to it
        self.polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="linear", count=100);
        self.polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        self.polymers = [self.polymer1, self.polymer2]
        self.box = hoomd.BoxDim(35);
        self.separation=dict(A=0.35, B=0.35)
        init.create_random_polymers(box=self.box, polymers=self.polymers, separation=self.separation);
        
        angle_data = globals.system_definition.getAngleData();
        angle_data.addAngle(hoomd.Angle(0, 0, 1, 2));
        import __main__;
        __main__.sorter.set_params(grid=8)
    
    # test to see that se can create an angle.harmonic
    def test_create(self):
        angle.harmonic();
        
    # test setting coefficients
    def test_set_coeff(self):
        harmonic = angle.harmonic();
        harmonic.set_coeff('angleA', k=1.0, t0=0.78125)
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);
        
    # test coefficient not set checking
    def test_set_coeff_fail(self):
        harmonic = angle.harmonic();
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        self.assertRaises(RuntimeError, run, 100);
    
    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

