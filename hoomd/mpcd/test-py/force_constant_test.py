# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd constant force
class mpcd_force_constant_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=1)
        snap.particles.position[:] = [[0.,-1.,1.]]
        snap.particles.velocity[:] = [[1.,-2.,3.]]
        self.s = mpcd.init.read_snapshot(snap)

    # test for initializing constant force field
    def test_init(self):
        # tuple
        f = mpcd.force.constant(F=(1.,2.1,-0.3))
        np.testing.assert_array_almost_equal(f.F, (1,2.1,-0.3))

        # list
        f = mpcd.force.constant(F=[-0.7,0,1])
        np.testing.assert_array_almost_equal(f.F, (-0.7,0,1))

        # numpy array
        f = mpcd.force.constant(F=np.array([1,2,3]))
        np.testing.assert_array_almost_equal(f.F, (1,2,3))

        # scalar is an error
        with self.assertRaises(ValueError):
            mpcd.force.constant(4.)

        # too short is an error
        with self.assertRaises(ValueError):
            mpcd.force.constant([1,2])

        # too long is an error
        with self.assertRaises(ValueError):
            mpcd.force.constant([1,2,3,4])

    # test for stepping with constant force
    # (this is also an implicit test that the base integrator is implementing the force correctly)
    def test_step(self):
        mpcd.integrator(dt=0.1)
        bulk = mpcd.stream.bulk(period=1)

        # run 1 step and check updated velocity
        f = mpcd.force.constant((1.,0.,-1.))
        bulk.set_force(f)
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], (1.1, -2.0, 2.9))
            np.testing.assert_array_almost_equal(snap.particles.position[0], (0.105, -1.2, 1.295))

        # remove force and stream freely
        bulk.remove_force()
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], (1.1, -2.0, 2.9))
            np.testing.assert_array_almost_equal(snap.particles.position[0], (0.215, -1.4, 1.585))

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
