# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd slit streaming geometry
class mpcd_stream_slit_test(unittest.TestCase):
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
        snap.particles.position[:] = [[4.95,-4.95,3.85],[0.,0.,-3.8]]
        snap.particles.velocity[:] = [[1.,-1.,1.],[-1.,-1.,-1.]]
        self.s = mpcd.init.read_snapshot(snap)

        mpcd.integrator(dt=0.1)

    # test creation can happen (with all parameters set)
    def test_create(self):
        mpcd.stream.slit(H=4., V=0.1, boundary="no_slip", period=2)

    # test for setting parameters
    def test_set_params(self):
        slit = mpcd.stream.slit(H=4.)
        self.assertAlmostEqual(slit.H, 4.)
        self.assertAlmostEqual(slit.V, 0.)
        self.assertEqual(slit.boundary, "no_slip")
        self.assertAlmostEqual(slit._geometry.H, 4.)
        self.assertAlmostEqual(slit._geometry.V, 0.)
        self.assertEqual(slit._geometry.boundary, mpcd._mpcd.boundary.no_slip)

        # change H and also ensure other parameters stay the same
        slit.set_params(H=2.)
        self.assertAlmostEqual(slit.H, 2.)
        self.assertAlmostEqual(slit.V, 0.)
        self.assertEqual(slit.boundary, "no_slip")
        self.assertAlmostEqual(slit._geometry.H, 2.)
        self.assertAlmostEqual(slit._geometry.V, 0.)
        self.assertEqual(slit._geometry.boundary, mpcd._mpcd.boundary.no_slip)

        # change V
        slit.set_params(V=0.1)
        self.assertAlmostEqual(slit.V, 0.1)
        self.assertAlmostEqual(slit._geometry.V, 0.1)

        # change BCs
        slit.set_params(boundary="slip")
        self.assertEqual(slit.boundary, "slip")
        self.assertEqual(slit._geometry.boundary, mpcd._mpcd.boundary.slip)

    # test for invalid boundary conditions being set
    def test_bad_boundary(self):
        slit = mpcd.stream.slit(H=4.)
        slit.set_params(boundary="no_slip")
        slit.set_params(boundary="slip")

        with self.assertRaises(ValueError):
            slit.set_params(boundary="invalid")

    # test basic stepping behavior with no slip boundary conditions
    def test_step_noslip(self):
        mpcd.stream.slit(H=4.)

        # take one step
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-4.95,4.95,3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,-1.,1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.1,-0.1,-3.9])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

        # take another step where one particle will now hit the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-4.95,4.95,3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.2,-0.2,-4.0])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

        # take another step, wrapping the second particle through the boundary
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [4.95,-4.95,3.85])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.1,-0.1,-3.9])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [1.,1.,1.])

    # test basic stepping behavior with slip boundary conditions
    def test_step_slip(self):
        mpcd.stream.slit(H=4., boundary="slip")

        # take one step
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-4.95,4.95,3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,-1.,1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.1,-0.1,-3.9])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

        # take another step where one particle will now hit the wall
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-4.85,4.85,3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,-1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.2,-0.2,-4.0])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,-1.])

        # take another step, wrapping the second particle through the boundary
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-4.75,4.75,3.85])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,-1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.3,-0.3,-3.9])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-1.,-1.,1.])

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
