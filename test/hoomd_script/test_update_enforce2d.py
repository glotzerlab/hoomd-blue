# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests for update.zero_momentum
class update_enforce2d_tests (unittest.TestCase):
    def setUp(self):
        print
        s = init.create_random(N=100, phi_p=0.05);
        s.dimensions = 2
        import __main__;
        __main__.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        update.enforce2d()
        run(100);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
