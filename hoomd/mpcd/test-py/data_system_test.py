# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import hoomd
from hoomd import mpcd

# unit tests for snapshots with mpcd particle data
class mpcd_snapshot(unittest.TestCase):
    def setUp(self):
        hoomd.context.initialize()
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=10.)))

    def test_set_params(self):
        s = mpcd.init.make_random(N=3, kT=1.0, seed=7)

        s.set_params(cell=1.5)

    def test_snapshot(self):
        s = mpcd.init.make_random(N=3, kT=1.0, seed=7)
        snap = s.take_snapshot()
        s.restore_snapshot(snap)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
