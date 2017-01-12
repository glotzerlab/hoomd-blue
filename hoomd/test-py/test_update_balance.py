# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

import hoomd
hoomd.context.initialize()
import unittest

## Dynamic load balancing tests
class load_balance_tests (unittest.TestCase):
    def setUp(self):
        snap = hoomd.data.make_snapshot(N=100, box=hoomd.data.boxdim(L=10), particle_types=['A'])
        hoomd.comm.decomposition(nx=1,ny=1,nz=2)
        hoomd.init.read_snapshot(snap)

    ## Test basic constructor succeeds
    def test_basic(self):
        hoomd.update.balance()

    ## Test for parameter setting in constructor and by set_params
    def test_set_params(self):
        lb = hoomd.update.balance(x=False, y=False, z=False, tolerance=1.05, maxiter=2, period=4, phase=1)
        if hoomd.context.current.decomposition is not None:
            lb.set_params(x=True, y=True, z=True, tolerance=0.95, maxiter=1)

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
