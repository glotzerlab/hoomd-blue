# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd sine force
class mpcd_force_sine_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot(N=1)
        snap.particles.position[:] = [[1.,-1.,0.]]
        snap.particles.velocity[:] = [[1.,-2.,2.]]
        self.s = mpcd.init.read_snapshot(snap)

    # test for initializing sine force field
    def test_init(self):
        # tuple
        f = mpcd.force.sine(F=2.,k=3.)
        self.assertAlmostEqual(f.F, 2.)
        self.assertAlmostEqual(f.k, 3.)

    # test for stepping with sine force
    # (this is also an implicit test that the base integrator is implementing the force correctly)
    def test_step(self):
        mpcd.integrator(dt=0.1)
        bulk = mpcd.stream.bulk(period=1)

        # run 1 step and check updated velocity
        k0 = 2.*np.pi/10.
        f = mpcd.force.sine(F=2., k=k0)
        bulk.set_force(f)
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.position[0], (1.1,-1.2,0.2))
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], (1.+0.5*0.1*2.*np.sin(k0*0.2), -2.0, 2.))

        # remove force and stream freely
        bulk.remove_force()
        hoomd.run(1)
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            np.testing.assert_array_almost_equal(snap.particles.velocity[0], (1.+0.5*0.1*2.*np.sin(k0*0.2), -2.0, 2.))

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
