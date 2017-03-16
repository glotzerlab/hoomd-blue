# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd integrator
class mpcd_integrator_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # set the decomposition in z for mpi builds
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nz=2)

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=1, box=hoomd.data.boxdim(L=20.)))

        # initialize the system from the starting snapshot
        mpcd.init.read_snapshot(mpcd.data.make_snapshot(N=1))

    # test basic creation
    def test_create(self):
        ig = mpcd.integrator(dt=0.001)
        self.assertTrue(type(ig._stream) is not None)
        self.assertEqual(ig._collide, None)
        self.assertEqual(hoomd.context.current.integrator, ig)

        mpcd.integrator(dt=0.001, period=50)
        mpcd.integrator(dt=0.005, aniso=True, period=5)

    # test setting available parameters
    def test_set_params(self):
        ig = mpcd.integrator(dt=0.001)
        self.assertAlmostEqual(ig.dt, 0.001)
        self.assertEqual(ig.aniso, None)

        # test changing dt
        ig.set_params(dt=0.005)
        self.assertAlmostEqual(ig.dt, 0.005)
        self.assertEqual(ig.aniso, None)

        # test changing aniso to False
        ig.set_params(aniso=False)
        self.assertAlmostEqual(ig.dt, 0.005)
        self.assertEqual(ig.aniso, False)

        # test chaning aniso to True
        ig.set_params(aniso=True)
        self.assertAlmostEqual(ig.dt, 0.005)
        self.assertEqual(ig.aniso, True)

    # test updating integration methods
    def test_update_methods(self):
        ig = mpcd.integrator(dt=0.001)
        ig.update_methods()

        # add an nve integrator
        md.integrate.nve(group=hoomd.group.all())
        ig.update_methods()

    # test a simple run command
    def test_run(self):
        mpcd.integrator(dt=0.001)
        hoomd.run(1)

    # test for error of system not initialized
    def test_not_init(self):
        hoomd.context.initialize()

        # calling before initialization must fail
        with self.assertRaises(RuntimeError):
            mpcd.integrator(dt=0.001)

        # calling after HOOMD initialization but not MPCD initialization must also fail
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=1, box=hoomd.data.boxdim(L=20.)))
        with self.assertRaises(RuntimeError):
            mpcd.integrator(dt=0.001)

        # now it must succeed
        mpcd.init.read_snapshot(mpcd.data.make_snapshot(N=1))
        mpcd.integrator(dt=0.001)

# unit tests for mpcd integrator
class mpcd_integrator_stream_test(unittest.TestCase):
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

        # step with a new integrator on a different period, which should step immediately
        mpcd.integrator(dt=0.05, period=4)
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
