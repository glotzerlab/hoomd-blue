# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for integrate.nve_rigid
class integrate_nve_rigid_tests (unittest.TestCase):
    def setUp(self):
        print
        sysdef = init.create_random(N=100, phi_p=0.05);
        for p in sysdef.particles:
            p.body = p.tag % 10

        sysdef.sysdef.getRigidData().initializeData()
        force.constant(fx=0.1, fy=0.1, fz=0.1)
                
    # tests basic creation of the integrater
    def test_basic(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.nve_rigid(all);
        run(1);
    
    def tearDown(self):
        init.reset();

# unit tests for integrate.nve_rigid w/o rigid bodies
class integrate_nve_rigid_nobody_tests (unittest.TestCase):
    def setUp(self):
        print
        sysdef = init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)

#FIXME
#    # test w/ empty group
#    def test_empty(self):
#        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
#        mode = integrate.mode_standard(dt=0.005);
#        nve = integrate.nve_rigid(group=empty)
#        run(1);

    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

