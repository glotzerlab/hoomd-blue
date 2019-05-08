# Copyright (c) 2009-2019 The Regents of the University of Michigan
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
        bulk = mpcd.stream.bulk(period=4)

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

        # trying to change the period on the wrong step should throw an error
        with self.assertRaises(RuntimeError):
            bulk.set_period(period=2)

        # running up to the next period should be allowed
        hoomd.run(3)
        bulk.set_period(period=2)

        # running once should now move half as far
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [1.8,-4.35,3.8])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-3.8,4.45,-1.8])

        # changing in between runs should fail
        with self.assertRaises(RuntimeError):
            bulk.set_period(1)
        hoomd.run(1)
        # cannot set period to a not equal multiple
        with self.assertRaises(RuntimeError):
            bulk.set_period(9)
        # but this should work since the timestep is 10
        bulk.set_period(5)

    def test_enable(self):
        mpcd.integrator(dt=0.1)
        bulk = mpcd.stream.bulk(period=1)
        self.assertTrue(bulk.enabled)
        self.assertEqual(hoomd.context.current.mpcd._stream, bulk)

        # ensure this is disabled
        bulk.disable()
        self.assertEqual(bulk.enabled, False)
        self.assertEqual(hoomd.context.current.mpcd._stream, None)

        bulk.enable()
        self.assertTrue(bulk.enabled)
        self.assertEqual(hoomd.context.current.mpcd._stream, bulk)

    # test for initialization order errors
    def test_init_errors(self):
        hoomd.context.initialize()

        # it is an error to make a collision rule without initializing first
        with self.assertRaises(RuntimeError):
            mpcd.stream.bulk()

        # it is an error to make a collision rule without initializing MPCD first
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=1, box=hoomd.data.boxdim(L=20.)))
        with self.assertRaises(RuntimeError):
            mpcd.stream.bulk()

        # OK, now it should go
        mpcd.init.read_snapshot(mpcd.data.make_snapshot(N=1))
        mpcd.stream.bulk()

        # creating another should raise an error
        with self.assertRaises(RuntimeError):
            mpcd.stream.bulk()

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
