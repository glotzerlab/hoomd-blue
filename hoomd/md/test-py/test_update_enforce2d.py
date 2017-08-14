# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# tests for update.zero_momentum
class update_enforce2d_tests (unittest.TestCase):
    def setUp(self):
        print
        s = init.create_lattice(lattice.sq(a=10),n=[10,10]);

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        md.update.enforce2d()
        run(100);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
