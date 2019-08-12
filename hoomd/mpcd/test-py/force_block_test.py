# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd block force
class mpcd_force_block_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=3)
        snap.particles.position[:] = [[0.,-1.,2.],[1.,1.,-2.],[0.,0.,0.]]
        self.s = mpcd.init.read_snapshot(snap)

    # test for initializing block force field
    def test_init(self):
        # default block size
        f = mpcd.force.block(F=0.5)
        self.assertAlmostEqual(f.F, 0.5)
        self.assertAlmostEqual(f.H, 2.5)
        self.assertAlmostEqual(f.w, 2.5)

        # specified block size
        f = mpcd.force.block(F=2.0, H=1.5, w=0.5)
        self.assertAlmostEqual(f.F, 2.0)
        self.assertAlmostEqual(f.H, 1.5)
        self.assertAlmostEqual(f.w, 0.5)

        # overlapping blocks is a warning, so should work
        f = mpcd.force.block(F=1.0, H=1.0, w=2.0)
        self.assertAlmostEqual(f.F, 1.0)
        self.assertAlmostEqual(f.H, 1.0)
        self.assertAlmostEqual(f.w, 2.0)

        # out of box is error
        with self.assertRaises(ValueError):
            mpcd.force.block(F=1.0, H=6.0, w=1.0)

        # slab out of box is error
        with self.assertRaises(ValueError):
            mpcd.force.block(F=1.0, H=4.0, w=2.0)

    # test for stepping with block force
    # (this is also an implicit test that the base integrator is implementing the force correctly)
    def test_step(self):
        mpcd.integrator(dt=0.1)
        bulk = mpcd.stream.bulk(period=1)

        # run 1 step and check updated velocity for all particles getting a force
        # velocities are reset at the end of the step
        f = mpcd.force.block(F=2.0)
        bulk.set_force(f)
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.velocity, ((0.2,0,0),(-0.2,0,0),(0.2,0,0)))
            snap.particles.velocity[:] = 0.
        self.s.restore_snapshot(snap)

        # run another step, but now the particle at the origin is outside the blocks
        # velocities are reset at the end of the step
        f = mpcd.force.block(F=2.0, H=2.1, w=0.2)
        bulk.set_force(f)
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.velocity, ((0.2,0,0),(-0.2,0,0),(0,0,0)))
            snap.particles.velocity[:] = 0.
        self.s.restore_snapshot(snap)

        # remove force and stream freely
        bulk.remove_force()
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.velocity, ((0,0,0),(0,0,0),(0,0,0)))

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
