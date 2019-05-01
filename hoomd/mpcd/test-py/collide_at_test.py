# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import md
from hoomd import mpcd

# unit tests for mpcd.collide.at
class mpcd_collide_at_test(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # default testing configuration
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=1, box=hoomd.data.boxdim(L=20.)))

        # initialize the system from the starting snapshot
        mpcd.init.read_snapshot(mpcd.data.make_snapshot(N=1))

        # create an integrator
        mpcd.integrator(dt=0.02)

    # test basic creation
    def test_create(self):
        at = mpcd.collide.at(seed=42, period=5, kT=1.0)
        self.assertEqual(at.enabled, True)
        self.assertEqual(hoomd.context.current.mpcd._collide, at)

        at.disable()
        self.assertEqual(at.enabled, False)
        self.assertEqual(hoomd.context.current.mpcd._collide, None)

        at.enable()
        self.assertEqual(at.enabled, True)
        self.assertEqual(hoomd.context.current.mpcd._collide, at)

        at.disable()
        mpcd.collide.at(seed=42, period=5, kT=hoomd.variant.linear_interp([[0,1.5],[10,2.0]]))

    # test for setting of embedded group
    def test_embed(self):
        group = hoomd.group.all()
        at = mpcd.collide.at(seed=42, period=5, kT=1.0, group=group)
        self.assertEqual(at.group, group)
        at.disable()

        at2 = mpcd.collide.at(seed=7, period=10, kT=1.0)
        at2.embed(group)
        self.assertEqual(at2.group, group)

    # test creation of multiple collision rules
    def test_multiple(self):
        # after a collision rule has been set, another cannot be created without
        # removing the first one
        at = mpcd.collide.at(seed=42, period=5, kT=1.0)
        with self.assertRaises(RuntimeError):
            mpcd.collide.at(seed=7, period=10, kT=1.0)

        # okay, now it should work
        at.disable()
        mpcd.collide.at(seed=7, period=10, kT=1.0)

    def test_set_params(self):
        at = mpcd.collide.at(seed=42, period=5, kT=1.0)
        self.assertEqual(at.shift, True)

        at.set_params(shift=False)
        self.assertEqual(at.shift, False)

        at.set_params(kT=2.0)
        at.set_params(kT=hoomd.variant.linear_interp([[0,1.5],[10,2.0]]))

    # test common initialization errors
    def test_init_errors(self):
        # clear out the system
        hoomd.context.initialize()

        # it is an error to make a collision rule without initializing first
        with self.assertRaises(RuntimeError):
            mpcd.collide.at(seed=42, period=5, kT=1.0)

        # it is an error to make a collision rule without initializing MPCD first
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=1, box=hoomd.data.boxdim(L=20.)))
        with self.assertRaises(RuntimeError):
            mpcd.collide.at(seed=42, period=5, kT=1.0)

        # OK, now it should go
        mpcd.init.read_snapshot(mpcd.data.make_snapshot(N=1))
        mpcd.collide.at(seed=42, period=5, kT=1.0)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
