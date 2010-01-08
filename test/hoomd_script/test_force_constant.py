# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests force.constant
class force_constant_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
    
    # test to see that se can create a force.constant
    def test_create(self):
        force.constant(fx=1.0, fy=0.5, fz=0.74);
        
    # test changing the force
    def test_change_force(self):
        const = force.constant(fx=1.0, fy=0.5, fz=0.74);
        const.set_force(fx=1.45, fy=0.25, fz=-0.1);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

