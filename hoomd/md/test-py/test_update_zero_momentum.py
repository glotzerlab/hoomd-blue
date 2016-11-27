# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
from hoomd import md;
context.initialize()
import unittest
import os

# tests for md.update.zero_momentum
class update_zero_momentum_tests (unittest.TestCase):
    def setUp(self):
        print
        deprecated.init.create_random(N=100, phi_p=0.05);

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        md.update.zero_momentum()
        run(100);

    # tests with phase
    def test_phase(self):
        md.update.zero_momentum(period=10, phase=0)
        run(100);

    # test variable periods
    def test_variable(self):
        md.update.zero_momentum(period = lambda n: n*100);
        run(100);

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
