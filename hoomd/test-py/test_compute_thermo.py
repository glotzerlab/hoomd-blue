# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
context.initialize()
import unittest
import os

# unit tests for analyze.msd
class compute_thermo_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

    # tests basic creation of the compute
    def test(self):
        typeA = group.type(name='typeA', type='A')
        compute.thermo(group=typeA);
        run(100);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
