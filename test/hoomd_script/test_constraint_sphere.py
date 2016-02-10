# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os

# test the constrain.sphere command
class constraint_sphere_tests (unittest.TestCase):
    def setUp(self):
        print
        self.sysdef = init.create_empty(N=2, box=data.boxdim(L=40), particle_types=['A']);
        self.sysdef.particles[0].position = (5,0,0);
        self.sysdef.particles[1].position = (-5,1,1);

        sorter.set_params(grid=8)

    def test_basic(self):
        all = group.all()
        constrain.sphere(group=all, P=(0,0,0), r=5)
        integrate.mode_standard(dt=0.005);
        integrate.langevin(group=all, T=1.2, seed=0);
        run(10);
        pos0 = self.sysdef.particles[0].position
        self.assertAlmostEqual(pos0[0]*pos0[0]+pos0[1]*pos0[1]+pos0[2]*pos0[2],5*5,1)
        pos1 = self.sysdef.particles[1].position
        self.assertAlmostEqual(pos1[0]*pos1[0]+pos1[1]*pos1[1]+pos1[2]*pos1[2],5*5,1)

    def test_error(self):
        all = group.all()
        self.assertRaises(RuntimeError, constrain.sphere, group=all, P=(0,0,0), r=10)

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
