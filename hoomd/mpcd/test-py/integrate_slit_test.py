# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import hoomd
from hoomd import md
from hoomd import mpcd
hoomd.context.initialize()
import unittest
import numpy as np

# unit tests for slit bounce back geometry
class integrate_slit_tests(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # set the decomposition in z for mpi builds
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nz=2)

        # default testing configuration
        snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=10.))
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [[4.95,-4.95,3.85],[0.,0.,-3.8]]
            snap.particles.velocity[:] = [[1.,-1.,1.],[-1.,-1.,-1.]]
            snap.particles.mass[:] = [1.,2.]
        self.s = hoomd.init.read_snapshot(snap)
        self.group = hoomd.group.all()

        md.integrate.mode_standard(dt=0.1)

    # test creation can happen (with all parameters set)
    def test_create(self):
        mpcd.integrate.slit(group=self.group, H=4., V=0.1, boundary="no_slip")

    # test for setting parameters
    def test_set_params(self):
        slit = mpcd.integrate.slit(group=self.group, H=4.)
        self.assertAlmostEqual(slit.H, 4.)
        self.assertAlmostEqual(slit.V, 0.)
        self.assertEqual(slit.boundary, "no_slip")
        self.assertAlmostEqual(slit.cpp_method.geometry.getH(), 4.)
        self.assertAlmostEqual(slit.cpp_method.geometry.getVelocity(), 0.)
        self.assertEqual(slit.cpp_method.geometry.getBoundaryCondition(), hoomd.mpcd._mpcd.boundary.no_slip)

        # change H and also ensure other parameters stay the same
        slit.set_params(H=2.)
        self.assertAlmostEqual(slit.H, 2.)
        self.assertAlmostEqual(slit.V, 0.)
        self.assertEqual(slit.boundary, "no_slip")
        self.assertAlmostEqual(slit.cpp_method.geometry.getH(), 2.)
        self.assertAlmostEqual(slit.cpp_method.geometry.getVelocity(), 0.)
        self.assertEqual(slit.cpp_method.geometry.getBoundaryCondition(), hoomd.mpcd._mpcd.boundary.no_slip)

        # change V
        slit.set_params(V=0.1)
        self.assertAlmostEqual(slit.V, 0.1)
        self.assertAlmostEqual(slit.cpp_method.geometry.getVelocity(), 0.1)

        # change BCs
        slit.set_params(boundary="slip")
        self.assertEqual(slit.boundary, "slip")
        self.assertEqual(slit.cpp_method.geometry.getBoundaryCondition(), hoomd.mpcd._mpcd.boundary.slip)

    # test for invalid boundary conditions being set
    def test_bad_boundary(self):
        slit = mpcd.integrate.slit(group=self.group, H=4.)
        slit.set_params(boundary="no_slip")
        slit.set_params(boundary="slip")

        with self.assertRaises(ValueError):
            slit.set_params(boundary="invalid")

    # test basic stepping behavior with no slip boundary conditions
    def test_step_noslip(self):
        mpcd.integrate.slit(group=self.group, H=4.)

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
        md.integrate.mode_standard(dt=0.3)
        mpcd.integrate.slit(group=self.group, H=4., boundary="no_slip", V=1.0)

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
            # same velocity along +x. It will bounce back in y and z to where it started.
            # (vx stays the same, and vy and vz flip.)
            np.testing.assert_array_almost_equal(snap.particles.position[0], [-4.75,-4.95,3.85])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.,1.,-1.])

            # the second particle has y and z velocities flip again, and since it started closer,
            # it moves relative to original position.
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.4,-0.1,-3.9])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [0.,1.,1.])

    # test basic stepping behavior with slip boundary conditions
    def test_step_slip(self):
        mpcd.integrate.slit(group=self.group, H=4., boundary="slip")

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

    # test for correct application of both verlet steps to particles away from boundary
    def test_accel(self):
        mpcd.integrate.slit(group=self.group, H=4.)
        md.force.constant(fx=2.,fy=-2.,fz=4.)

        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [[0,0,0],[0,0,0]]
        self.s.restore_snapshot(snap)

        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], [0.11,-0.11,0.12])
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], [1.2,-1.2,1.4])
            np.testing.assert_array_almost_equal(snap.particles.position[1], [-0.095,-0.105,-0.09])
            np.testing.assert_array_almost_equal(snap.particles.velocity[1], [-0.9,-1.1,-0.8])

    # test that setting the slit size too large raises an error
    def test_validate_box(self):
        # initial configuration is invalid
        slit = mpcd.integrate.slit(group=self.group, H=10.)
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
        slit = mpcd.integrate.slit(group=self.group, H=3.8)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

        slit.set_params(H=3.85)
        hoomd.run(1)

    def test_aniso(self):
        md.integrate.mode_standard(dt=0.1, aniso=True)
        slit = mpcd.integrate.slit(group=self.group, H=4.)
        with self.assertRaises(RuntimeError):
            hoomd.run(1)

    def tearDown(self):
        del self.s, self.group

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
