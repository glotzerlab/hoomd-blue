# Copyright (c) 2009-2019 The Regents of the University of Michigan
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
        self.assertEqual(hoomd.context.current.integrator, ig)

        mpcd.integrator(dt=0.001)
        mpcd.integrator(dt=0.005, aniso=True)

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

    # test possible errors with the collision period and streaming period with the integrator
    def test_bad_period(self):
        ig = mpcd.integrator(dt=0.001)
        bulk = mpcd.stream.bulk(period=5)

        # period cannot be less than integrator's period
        srd = mpcd.collide.srd(seed=42, period=1, angle=130.)
        with self.assertRaises(ValueError):
            ig.update_methods()
        srd.disable()

        # being equal is OK
        srd = mpcd.collide.srd(seed=42, period=5, angle=130.)
        ig.update_methods()
        srd.disable()

        # period being greater but not a multiple is also an error
        srd = mpcd.collide.srd(seed=42, period=7, angle=130.)
        with self.assertRaises(ValueError):
            ig.update_methods()
        srd.disable()

        # being greater and a multiple is OK
        srd = mpcd.collide.srd(seed=42, period=10, angle=130.)
        ig.update_methods()

        # using set_period interface should also cause a problem now though
        bulk.set_period(7)
        with self.assertRaises(ValueError):
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

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
