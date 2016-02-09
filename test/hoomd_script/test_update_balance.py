
# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest

## Dynamic load balancing tests
class load_balance_tests (unittest.TestCase):
    def setUp(self):
        if comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()
            comm.decomposition(nx=2,ny=2,nz=2)
            init.create_random(N=100, phi_p=0.05)

    ## Test basic constructor succeeds
    def test_basic(self):
        if comm.get_num_ranks() > 1:
            update.balance()

    ## Test for parameter setting in constructor and by set_params
    def test_set_params(self):
        if comm.get_num_ranks() > 1:
            lb = update.balance(x=False, y=False, z=False, tolerance=1.05, maxiter=2, period=4, phase=1)
            lb.set_params(x=True, y=True, z=True, tolerance=0.95, maxiter=1)

    def tearDown(self):
        if comm.get_num_ranks() > 1:
            init.reset()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
