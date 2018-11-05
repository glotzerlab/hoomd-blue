from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import os
import numpy

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# Tests to ensure that all particle types can be created

class polyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=2, box=data.boxdim(L=25, dimensions=3), particle_types=['A'])

        self.mc = hpmc.integrate.polyhedron(seed=10,d=0,a=0);

    # special case of a spheropolyhedron: a sphere
    def test_two_spheres(self):
        # two overlapping spheres
        self.system.particles[0].position = (-2.94709386, -0.21256315, -5.01176827)
        self.system.particles[0].orientation = (1,0,0,0)
        self.system.particles[1].position = (-2.7780668,  0.641089889, -5.32930423)
        self.system.particles[1].orientation = (1,0,0,0)

        self.mc.shape_param.set('A',sweep_radius=0.5,faces=[[0]], vertices=[[0,0,0]])
        run(1) # to satisy count_overlaps()

        # verify that the spheres are overlapping
        self.assertEqual(self.mc.count_overlaps(), 1);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
