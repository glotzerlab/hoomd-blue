# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os

# tests for update.sorter
class update_sorter_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

    # test set_params
    def test_set_params(self):

        context.current.sorter.set_params(grid=20);

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
