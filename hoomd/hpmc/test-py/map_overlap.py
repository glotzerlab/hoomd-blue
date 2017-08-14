from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init
from hoomd import hpmc

import unittest
import numpy as np

context.initialize()

class map_overlaps_test(unittest.TestCase):

    def setUp(self):
        snap = data.make_snapshot(3, data.boxdim(Lx = 3.5, Ly = 1.5, Lz = 1.5))

        #Setup a set of positions which we can easily see the overlaps for
        snap.particles.position[0] = [-.9, 0.0, 0.0]
        snap.particles.position[1] = [0.0, 0.0, 0.0]
        snap.particles.position[2] = [1.0, 0.25, 0.0]

        self.system = init.read_snapshot(snap)
        self.mc = hpmc.integrate.sphere(seed=123)

    def tearDown(self):
        context.initialize()

    def test_single_overlap(self):
        self.mc.shape_param.set('A', diameter=1.0)
        overlap_map = np.asarray(self.mc.map_overlaps())
        for i in range(3):
            for j in range(3):
                if (i==0 and j==1):
                    self.assertTrue(overlap_map[i][j])
                else:
                    self.assertFalse(overlap_map[i][j])

    def test_double_overlap(self):
        self.mc.shape_param.set('A', diameter=1.1)
        overlap_map = np.asarray(self.mc.map_overlaps())
        for i in range(3):
            for j in range(3):
                if (i==0 and j==1):
                    self.assertTrue(overlap_map[i][j])
                elif (i==1 and j==2):
                    self.assertTrue(overlap_map[i][j])
                else:
                    self.assertFalse(overlap_map[i][j])

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
