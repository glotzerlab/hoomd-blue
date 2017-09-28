# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd integrator
class mpcd_stream_bulk_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # set the decomposition in z for mpi builds
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nz=2)

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=2)
        snap.particles.position[:] = [[1.,4.85,3.],[-3.,-4.75,-1.]]
        snap.particles.velocity[:] = [[1.,1.,1.],[-1.,-1.,-1.]]
        self.s = mpcd.init.read_snapshot(snap)

    # test basic stepping behavior
    def test_step(self):
        mpcd.integrator(dt=0.1)
        mpcd.stream.bulk(period=1)

        # take one step
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [1.1,4.95,3.1])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-3.1,-4.85,-1.1])

        # take another step, wrapping the first particle through the boundary
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [1.2,-4.95,3.2])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-3.2,-4.95,-1.2])

        # take another step, wrapping the second particle through the boundary
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [1.3,-4.85,3.3])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-3.3,4.95,-1.3])

    # test that streaming can proceed periodically
    def test_period(self):
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = [1.3,-4.85,3.3]
            snap.particles.position[1] = [-3.3,4.95,-1.3]
        self.s.restore_snapshot(snap)

        mpcd.integrator(dt=0.05)
        mpcd.stream.bulk(period=4)

        # first step should go
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [1.5,-4.65,3.5])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-3.5,4.75,-1.5])

        # running again should not move the particles since we haven't hit next period
        hoomd.run(3)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [1.5,-4.65,3.5])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-3.5,4.75,-1.5])

        # but one more step should move them again
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [1.7,-4.45,3.7])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-3.7,4.55,-1.7])

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
