# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

import itertools
import unittest
import numpy as np
from hoomd.dem.utils import *

np.random.seed(10)

class utils(unittest.TestCase):
    def test_mass_properties_2d(self, tries=32):
        """Run some tests for the massProperties function in dem.utils for
        2D shapes. Takes a tunable floating point tolerance and a number
        of random trials to test. Returns True if all tests pass"""

        nverts = np.random.randint(3, 16)
        verts = np.random.uniform(-3, 3, (nverts, 2))

        (mass0, com0, moment0) = massProperties(verts)
        verts -= com0

        for _ in range(tries):
            theta = np.random.uniform(0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

            verts = verts.dot(rot)
            (mass, com, moment) = massProperties(verts)

            self.assertAlmostEqual(mass, mass)
            for (est, real) in zip(com, [0, 0]):
                self.assertAlmostEqual(est, real)
            self.assertAlmostEqual(moment0[5], moment[5])

    def test_mass_properties_3d(self, tries=32):
        try:
            from scipy.spatial import ConvexHull as _
        except ImportError:
            msg = 'Convex hull not available, skipping hoomd.dem.utils.convexHull test'
            raise unittest.SkipTest(msg)

        for _ in range(tries):
            # take a random displacement; make sure that convexHull works
            # even for non-centered inputs
            delta = np.random.uniform(-2, 2, 3)
            massFactor = np.random.uniform(.5, 1.5)

            vertices = np.array(list(itertools.product(*(3*[[-1., 1.]]))))
            (vertices, faces) = convexHull(vertices)

            # convexHull centers its output, so add on another random
            # vector for testing purposes
            delta = np.random.uniform(-2, 2, 3)
            (mass, com, moment) = massProperties(vertices + delta, faces, factor=massFactor)

            self.assertAlmostEqual(mass, 8*massFactor)
            for (est, real) in zip(com, delta):
                self.assertAlmostEqual(est, real)

            trueMoment = [16./3*massFactor, 0, 0, 16./3*massFactor, 0, 16./3*massFactor]
            for (est, real) in zip(moment, trueMoment):
                self.assertAlmostEqual(est, real)


        for _ in range(tries):
            nverts = np.random.randint(4, 16)
            verts = np.random.uniform(-1, 1, (nverts, 3))

            (verts, faces) = convexHull(verts)

            verts = center(verts, faces)

            (mass, com, _) = massProperties(verts, faces)

            self.assertGreaterEqual(mass, 0)

            for est in com:
                self.assertAlmostEqual(est, 0)

if __name__ == '__main__':
    unittest.main(argv = ['test_utils.py', '-v']);
