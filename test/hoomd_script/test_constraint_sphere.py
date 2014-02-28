# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# test the constrain.sphere command
class constraint_sphere_tests (unittest.TestCase):
    def setUp(self):
        print
        sysdef = init.create_empty(N=2, box=(40,40,40), n_particle_types=1);
        sysdef.particles[0].position = (5,0,0);
        sysdef.particles[1].position = (-5,1,1);

        sorter.set_params(grid=8)

    def test_basic(self):
        all = group.all()
        constrain.sphere(group=all, P=(0,0,0), r=5)
        integrate.mode_standard(dt=0.005);
        integrate.bdnvt(group=all, T=1.2);
        run(10);

    def test_error(self):
        all = group.all()
        self.assertRaises(RuntimeError, constrain.sphere, group=all, P=(0,0,0), r=10)

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
