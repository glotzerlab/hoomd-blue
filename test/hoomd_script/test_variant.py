# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests for variant types
class variant_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests creation of the constant variant
    def test_const(self):
        v = variant._constant(5)
        self.assertEqual(5.0, v.cpp_variant.getValue(0))
        self.assertEqual(5.0, v.cpp_variant.getValue(100000))
        self.assertEqual(5.0, v.cpp_variant.getValue(5000))
        self.assertEqual(5.0, v.cpp_variant.getValue(40))
        self.assertEqual(5.0, v.cpp_variant.getValue(50))

    # tests a simple linear variant
    def test_linear_interp(self):
        v = variant.linear_interp(points = [(0, 10), (100, 20)]);
        self.assertEqual(15.0, v.cpp_variant.getValue(50));
        self.assertEqual(10.0, v.cpp_variant.getValue(0));
        self.assertEqual(20.0, v.cpp_variant.getValue(100));
        self.assertEqual(20.0, v.cpp_variant.getValue(1000));

    # test the setup helper
    def setup_variant_input_test(self):
        v = variant._setup_variant_input(55);
        self.assertEqual(55.0, v.cpp_variant.getValue(0));

        v = variant._setup_variant_input(variant.linear_interp(points = [(0, 10), (100, 20)]));
        self.assertEqual(15.0, v.cpp_variant.getValue(50));

    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

