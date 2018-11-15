from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init
from hoomd import hpmc

import unittest
import numpy as np

import math

context.initialize()

class map_overlaps_test(unittest.TestCase):

    def setUp(self):
        snap = data.make_snapshot(N=0,box=data.boxdim(Lx = 10, Ly = 10, Lz = 10))

        self.system = init.read_snapshot(snap)
        self.mc = hpmc.integrate.convex_polyhedron(seed=123)

    def tearDown(self):
        context.initialize()

    def test_single_overlap(self):
        # two cubes, with one rotated by 45 degrees around two axes
        qi = [1,0,0,0]
        #alpha = math.pi/4.0;
        #q1 = (math.cos(alpha/2),math.sin(alpha/2),0,0)
        #q2 = (math.cos(alpha/2),0,0,math.sin(alpha/2))
        #import rowan
        # qj = rowan.multiply(q2,q1)
        qj = [0.85355339, 0.35355339, 0.14644661, 0.35355339]
        self.mc.shape_param.set('A', vertices = [(-0.5,-0.5,-0.5), (0.5,-0.5,-0.5), (0.5,0.5,-0.5), (-0.5,0.5,-0.5),
            (-0.5,-0.5,0.5), (0.5,-0.5,0.5), (0.5,0.5,0.5), (-0.5,0.5,0.5)])

        # not overlapping
        rij = np.array((-0.2,-1.4,0))
        self.assertFalse(self.mc.test_overlap('A','A',rij, qi, qj))
        self.assertFalse(self.mc.test_overlap('A','A',-rij, qj, qi))

        # overlapping point-face
        rij = np.array((0,1.2,0))
        self.assertTrue(self.mc.test_overlap('A','A',rij, qi, qj))
        self.assertTrue(self.mc.test_overlap('A','A',-rij, qj, qi))

    def test_image_overlap(self):
        qi = [1,0,0,0]
        qj = [0.85355339, 0.35355339, 0.14644661, 0.35355339]

        self.mc.shape_param.set('A', vertices = [(-0.5,-0.5,-0.5), (0.5,-0.5,-0.5), (0.5,0.5,-0.5), (-0.5,0.5,-0.5),
            (-0.5,-0.5,0.5), (0.5,-0.5,0.5), (0.5,0.5,0.5), (-0.5,0.5,0.5)])

        # not overlapping, excluding the self image
        rij = np.array((0,1.2,0))
        self.assertFalse(self.mc.test_overlap('A','A',rij, qi, qj, exclude_self=True))
        self.assertFalse(self.mc.test_overlap('A','A',-rij, qj, qi, exclude_self=True))

        # not overlapping, one image apart
        rij = np.array((10-0.2,-1.4,0))
        self.assertFalse(self.mc.test_overlap('A','A',rij, qi, qj))
        self.assertFalse(self.mc.test_overlap('A','A',-rij, qj, qi))

        self.assertFalse(self.mc.test_overlap('A','A',rij, qi, qj, use_images=False))
        self.assertFalse(self.mc.test_overlap('A','A',-rij, qj, qi, use_images=False))


        # overlapping, point-face, one image apart
        rij =  np.array((10,1.2,0))
        self.assertTrue(self.mc.test_overlap('A','A',rij, qi, qj))
        self.assertTrue(self.mc.test_overlap('A','A',-rij, qj, qi))

        self.assertFalse(self.mc.test_overlap('A','A',rij, qi, qj, use_images=False))
        self.assertFalse(self.mc.test_overlap('A','A',-rij, qj, qi, use_images=False))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
