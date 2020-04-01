# -*- coding: iso-8859-1 -*-
# Maintainer: pschoenh

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os

# tests for update.hypersphere_resize
class update_hypersphere_resize_tests (unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=100, hypersphere=data.hypersphere(R=20),particle_types=['A'])
        self.s = init.read_snapshot(snap)

    # tests basic creation of the updater
    def test(self):
        update.hypersphere_resize(R = variant.linear_interp([(0, 20), (99, 50)]))
        run(100);
        self.assertAlmostEqual(self.s.hypersphere.R,50)

    # tests with phase
    def test_phase(self):
        update.hypersphere_resize(R = variant.linear_interp([(0, 20), (1e6, 50)]), period=10, phase=0)
        run(100);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

