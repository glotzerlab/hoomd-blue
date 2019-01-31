# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import mpcd

# unit tests for snapshots with mpcd particle data
class mpcd_init_make_random(unittest.TestCase):
    def setUp(self):
        hoomd.context.initialize()
        # initialize an empty snapshot
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

    def test_init(self):
        s = mpcd.init.make_random(N=3, kT=1.0, seed=7)

        # check number of particles
        self.assertEqual(s.particles.N_global, 3)
        if hoomd.comm.get_num_ranks() > 1:
            if hoomd.comm.get_rank() == 0:
                self.assertEqual(s.particles.N, 2)
            else:
                self.assertEqual(s.particles.N, 1)

        # check tags
        if hoomd.comm.get_num_ranks() > 1:
            if hoomd.comm.get_rank() == 0:
                self.assertEqual(s.particles.getTag(0), 0)
                self.assertEqual(s.particles.getTag(1), 1)
            else:
                self.assertEqual(s.particles.getTag(0), 2)
        else:
            self.assertEqual(s.particles.getTag(0), 0)
            self.assertEqual(s.particles.getTag(1), 1)
            self.assertEqual(s.particles.getTag(2), 2)

        # check default type creation
        self.assertEqual(s.particles.n_types, 1)
        self.assertEqual(s.particles.getNameByType(0), "A")

        # check default mass
        self.assertEqual(s.particles.mass, 1.0)

    def test_random(self):
        s = mpcd.init.make_random(N=100000, kT=0.5, seed=7)
        snap = s.take_snapshot()

        if hoomd.comm.get_rank() == 0:
            # histogram particles long x, y, and z and check uniform with loose tol
            pos = snap.particles.position
            hist,_ = np.histogram(pos[:,0], bins=10, range=(-5.,5.))
            np.testing.assert_allclose(hist, 10000., rtol=0.05)
            hist,_ = np.histogram(pos[:,1], bins=10, range=(-5.,5.))
            np.testing.assert_allclose(hist, 10000., rtol=0.05)
            hist,_ = np.histogram(pos[:,2], bins=10, range=(-5.,5.))
            np.testing.assert_allclose(hist, 10000., rtol=0.05)

            # check velocities are distributed OK using loose tolerance on mean and variance
            vel = snap.particles.velocity
            vel = np.reshape(vel, (3*snap.particles.N, 1))
            self.assertAlmostEqual(np.mean(vel), 0.0, places=5)
            # sigma^2 = kT / m
            self.assertAlmostEqual(np.mean(vel**2), 0.5, places=2)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
