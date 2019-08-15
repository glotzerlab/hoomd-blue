# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd slit pore streaming geometry
class mpcd_stream_slit_pore_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # set the decomposition in z for mpi builds
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nz=2)

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=8)
        snap.particles.position[:] = [[-3.05,-4,-4.11],
                                      [ 3.05, 4, 4.11],
                                      [-3.05,-2, 4.11],
                                      [ 3.05, 2,-4.11],
                                      [ 0   , 0, 3.95],
                                      [ 0   , 0,-3.95],
                                      [ 3.03, 0, -3.98],
                                      [ 3.02, 0, -3.97]]
        snap.particles.velocity[:] = [[ 1.,-1., 1.],
                                      [-1., 1.,-1.],
                                      [ 1., 0.,-1.],
                                      [-1., 0., 1.],
                                      [ 0., 0., 1.],
                                      [ 0., 0.,-1.],
                                      [-1., 0.,-1.],
                                      [-1., 0.,-1.]]
        self.s = mpcd.init.read_snapshot(snap)

        mpcd.integrator(dt=0.1)

    # test creation can happen (with all parameters set)
    def test_create(self):
        mpcd.stream.slit_pore(H=4., L=2., boundary="no_slip", period=2)

    # test for setting parameters
    def test_set_params(self):
        slit_pore = mpcd.stream.slit_pore(H=4., L=2.)
        self.assertAlmostEqual(slit_pore.H, 4.)
        self.assertAlmostEqual(slit_pore.L, 2.)
        self.assertEqual(slit_pore.boundary, "no_slip")
        self.assertAlmostEqual(slit_pore._cpp.geometry.getH(), 4.)
        self.assertAlmostEqual(slit_pore._cpp.geometry.getL(), 2.)
        self.assertEqual(slit_pore._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change H and also ensure other parameters stay the same
        slit_pore.set_params(H=2.)
        self.assertAlmostEqual(slit_pore.H, 2.)
        self.assertAlmostEqual(slit_pore.L, 2.)
        self.assertEqual(slit_pore.boundary, "no_slip")
        self.assertAlmostEqual(slit_pore._cpp.geometry.getH(), 2.)
        self.assertAlmostEqual(slit_pore._cpp.geometry.getL(), 2.)
        self.assertEqual(slit_pore._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.no_slip)

        # change L
        slit_pore.set_params(L=1.)
        self.assertAlmostEqual(slit_pore.L, 1.)
        self.assertAlmostEqual(slit_pore._cpp.geometry.getL(), 1.)

        # change BCs
        slit_pore.set_params(boundary="slip")
        self.assertEqual(slit_pore.boundary, "slip")
        self.assertEqual(slit_pore._cpp.geometry.getBoundaryCondition(), mpcd._mpcd.boundary.slip)

    # test for invalid boundary conditions being set
    def test_bad_boundary(self):
        slit_pore = mpcd.stream.slit_pore(H=4.,L=2.)
        slit_pore.set_params(boundary="no_slip")
        slit_pore.set_params(boundary="slip")

        with self.assertRaises(ValueError):
            slit_pore.set_params(boundary="invalid")

    # test basic stepping behavior with no slip boundary conditions
    def test_step_noslip(self):
        mpcd.stream.slit_pore(H=4.,L=3.)

        # take one step, and everything should collide and bounce back
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-3.05,-4,-4.11])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1., 1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [ 3.05, 4, 4.11])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [ 1.,-1., 1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], [-3.05,-2, 4.11])
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], [-1., 0., 1.])
            np.testing.assert_array_almost_equal(snap.particles.position[3], [ 3.05, 2,-4.11])
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], [ 1., 0.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[4], [ 0, 0, 3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[4], [ 0, 0, -1.])
            np.testing.assert_array_almost_equal(snap.particles.position[5], [ 0, 0,-3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[5], [ 0, 0, 1.])
            # hits z = -4 after 0.02, then reverses. x is 3.01, so reverses to 3.09
            np.testing.assert_array_almost_equal(snap.particles.position[6], [ 3.09, 0,-3.92])
            np.testing.assert_array_almost_equal(snap.particles.velocity[6], [ 1, 0, 1])
            # hits x = 3 after 0.02, then reverses. z is -3.99, so reverses to -3.91
            np.testing.assert_array_almost_equal(snap.particles.position[7], [ 3.08, 0,-3.91])
            np.testing.assert_array_almost_equal(snap.particles.velocity[7], [ 1, 0, 1])

        # take another step where nothing hits now
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-3.15,-3.9,-4.21])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [ 3.15, 3.9, 4.21])
            np.testing.assert_array_almost_equal(snap.particles.position[2], [-3.15,-2, 4.21])
            np.testing.assert_array_almost_equal(snap.particles.position[3], [ 3.15, 2,-4.21])
            np.testing.assert_array_almost_equal(snap.particles.position[4], [ 0, 0, 3.85])
            np.testing.assert_array_almost_equal(snap.particles.position[5], [ 0, 0,-3.85])
            np.testing.assert_array_almost_equal(snap.particles.position[6], [ 3.19, 0,-3.82])
            np.testing.assert_array_almost_equal(snap.particles.position[7], [ 3.18, 0,-3.81])

    # test basic stepping behavior with slip boundary conditions
    def test_step_slip(self):
        mpcd.stream.slit_pore(H=4.,L=3.,boundary="slip")

        # take one step, and everything should collide and bounce back
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-3.05,-4.1,-4.01])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [-1.,-1.,1.])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [ 3.05, 4.1, 4.01])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [ 1., 1.,-1.])
            np.testing.assert_array_almost_equal(snap.particles.position[2], [-3.05,-2, 4.01])
            np.testing.assert_array_almost_equal(snap.particles.velocity[2], [-1., 0., -1.])
            np.testing.assert_array_almost_equal(snap.particles.position[3], [ 3.05, 2,-4.01])
            np.testing.assert_array_almost_equal(snap.particles.velocity[3], [ 1., 0., 1.])
            np.testing.assert_array_almost_equal(snap.particles.position[4], [ 0, 0, 3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[4], [ 0, 0, -1.])
            np.testing.assert_array_almost_equal(snap.particles.position[5], [ 0, 0,-3.95])
            np.testing.assert_array_almost_equal(snap.particles.velocity[5], [ 0, 0, 1.])
            # hits z = -4 after 0.02, then reverses. x is not touched because slip
            np.testing.assert_array_almost_equal(snap.particles.position[6], [ 2.93, 0,-3.92])
            np.testing.assert_array_almost_equal(snap.particles.velocity[6], [-1, 0, 1])
            # hits x = 3 after 0.02, then reverses. z is not touched because slip
            np.testing.assert_array_almost_equal(snap.particles.position[7], [ 3.08, 0,-4.07])
            np.testing.assert_array_almost_equal(snap.particles.velocity[7], [ 1, 0,-1])

        # take another step where nothing hits now
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-3.15,-4.2,-3.91])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [ 3.15, 4.2, 3.91])
            np.testing.assert_array_almost_equal(snap.particles.position[2], [-3.15,-2, 3.91])
            np.testing.assert_array_almost_equal(snap.particles.position[3], [ 3.15, 2,-3.91])
            np.testing.assert_array_almost_equal(snap.particles.position[4], [ 0, 0, 3.85])
            np.testing.assert_array_almost_equal(snap.particles.position[5], [ 0, 0,-3.85])
            np.testing.assert_array_almost_equal(snap.particles.position[6], [ 2.83, 0,-3.82])
            np.testing.assert_array_almost_equal(snap.particles.position[7], [ 3.18, 0,-4.17])

    # test that setting the slit size too large raises an error
    def test_validate_box(self):
        # zero velocities to stop particles moving during testing
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            snap.particles.velocity[:] = 0.
        self.s.restore_snapshot(snap)

        # initial configuration is invalid
        slit_pore = mpcd.stream.slit_pore(H=10.,L=2.)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        slit_pore.set_params(H=4.,L=10.)
        with self.assertRaises(RuntimeError):
            hoomd.run(2)

        # now it should be valid
        slit_pore.set_params(H=4.,L=3.)
        hoomd.run(3)

        # make sure we can invalidate it again
        slit_pore.set_params(H=10.,L=10.)
        with self.assertRaises(RuntimeError):
            hoomd.run(4)

    # test that particles out of bounds can be caught
    def test_out_of_bounds(self):
        slit_pore = mpcd.stream.slit_pore(H=3.8, L=3.)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        slit_pore.set_params(H=4)
        hoomd.run(1)

        slit_pore.set_params(L=3.5)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    # test that virtual particle filler can be attached, removed, and updated
    def test_filler(self):
        # initialization of a filler
        slit_pore = mpcd.stream.slit_pore(H=4.,L=3.)
        slit_pore.set_filler(density=5., kT=1.0, seed=42, type='A')
        self.assertTrue(slit_pore._filler is not None)

        # run should be able to setup the filler, although this all happens silently
        hoomd.run(1)

        # changing the geometry should still be OK with a run
        slit_pore.set_params(L=2.5)
        hoomd.run(1)

        # changing filler should be allowed
        slit_pore.set_filler(density=10., kT=1.5, seed=7)
        self.assertTrue(slit_pore._filler is not None)
        hoomd.run(1)

        # assert an error is raised if we set a bad particle type
        with self.assertRaises(RuntimeError):
            slit_pore.set_filler(density=5., kT=1.0, seed=42, type='B')

        # assert an error is raised if we set a bad density
        with self.assertRaises(RuntimeError):
            slit_pore.set_filler(density=-1.0, kT=1.0, seed=42)

        # removing the filler should still allow a run
        slit_pore.remove_filler()
        self.assertTrue(slit_pore._filler is None)
        hoomd.run(1)

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
