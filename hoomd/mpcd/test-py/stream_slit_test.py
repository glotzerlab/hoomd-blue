# Copyright (c) 2009-2019 The Regents of the University of Michigan
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
        self.assertAlmostEqual(slit._cpp.geometry.getH(), 4.)
        self.assertAlmostEqual(slit._cpp.geometry.getVelocity(), 0.)
        self.assertEqual(slit._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change H and also ensure other parameters stay the same
        slit.set_params(H=2.)
        self.assertAlmostEqual(slit.H, 2.)
        self.assertAlmostEqual(slit.V, 0.)
        self.assertEqual(slit.boundary, "no_slip")
        self.assertAlmostEqual(slit._cpp.geometry.getH(), 2.)
        self.assertAlmostEqual(slit._cpp.geometry.getVelocity(), 0.)
        self.assertEqual(slit._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change V
        slit.set_params(V=0.1)
        self.assertAlmostEqual(slit.V, 0.1)
        self.assertAlmostEqual(slit._cpp.geometry.getVelocity(), 0.1)

        # change BCs
        slit.set_params(boundary="slip")
        self.assertEqual(slit.boundary, "slip")
        self.assertEqual(slit._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.slip)

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

    def test_step_moving_wall(self):
        mpcd.stream.slit(H=4., boundary="no_slip", V=1.0, period=3)

        # change velocity of lower particle so it is translating relative to wall
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            snap.particles.velocity[1] = [-2.,-1.,-1.]
        self.s.restore_snapshot(snap)

        # run one step and check bounce back of particles
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            # the first particle is matched exactly to the wall speed, and so it will translate at
            # same velocity along +x for 3 steps. It will bounce back in y and z to where it started.
            # (vx stays the same, and vy and vz flip.)
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-4.75,-4.95,3.85])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,1.,-1.])

            # the second particle has y and z velocities flip again, and since it started closer,
            # it moves relative to original position.
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.4,-0.1,-3.9])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [0.,1.,1.])

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

    # test that setting the slit size too large raises an error
    def test_validate_box(self):
        # initial configuration is invalid
        slit = mpcd.stream.slit(H=10.)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        # now it should be valid
        slit.set_params(H=4.)
        hoomd.run(2)

        # make sure we can invalidate it again
        slit.set_params(H=10.)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    # test that particles out of bounds can be caught
    def test_out_of_bounds(self):
        slit = mpcd.stream.slit(H=3.8)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        slit.set_params(H=3.85)
        hoomd.run(1)

    # test that virtual particle filler can be attached, removed, and updated
    def test_filler(self):
        # initialization of a filler
        slit = mpcd.stream.slit(H=4.)
        slit.set_filler(density=5., kT=1.0, seed=42, type='A')
        self.assertTrue(slit._filler is not None)

        # run should be able to setup the filler, although this all happens silently
        hoomd.run(1)

        # changing the geometry should still be OK with a run
        slit.set_params(V=1.0)
        hoomd.run(1)

        # changing filler should be allowed
        slit.set_filler(density=10., kT=1.5, seed=7)
        self.assertTrue(slit._filler is not None)
        hoomd.run(1)

        # assert an error is raised if we set a bad particle type
        with self.assertRaises(RuntimeError):
            slit.set_filler(density=5., kT=1.0, seed=42, type='B')

        # assert an error is raised if we set a bad density
        with self.assertRaises(RuntimeError):
            slit.set_filler(density=-1.0, kT=1.0, seed=42)

        # removing the filler should still allow a run
        slit.remove_filler()
        self.assertTrue(slit._filler is None)
        hoomd.run(1)

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
