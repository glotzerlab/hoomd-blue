# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os

# tests for variant types
class variant_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

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

    # test the zero option on linear_interp
    def test_linear_interp(self):
        run(1000)

        v = variant.linear_interp(points = [(0, 10), (100, 20)], zero='now');
        self.assertEqual(15.0, v.cpp_variant.getValue(1050));
        self.assertEqual(10.0, v.cpp_variant.getValue(1000));
        self.assertEqual(20.0, v.cpp_variant.getValue(1100));
        self.assertEqual(20.0, v.cpp_variant.getValue(2000));

        v2 = variant.linear_interp(points = [(0, 10), (100, 20)], zero=500);
        self.assertEqual(15.0, v2.cpp_variant.getValue(550));
        self.assertEqual(10.0, v2.cpp_variant.getValue(500));
        self.assertEqual(20.0, v2.cpp_variant.getValue(600));
        self.assertEqual(20.0, v2.cpp_variant.getValue(1500));

        # test input sanitization
        v2 = variant.linear_interp(points = [(0.0, 10.0), (100.0, 20.0)], zero=5.0e2);
        self.assertEqual(15.0, v2.cpp_variant.getValue(550));
        self.assertEqual(10.0, v2.cpp_variant.getValue(500));
        self.assertEqual(20.0, v2.cpp_variant.getValue(600));
        self.assertEqual(20.0, v2.cpp_variant.getValue(1500));

    # test the setup helper
    def setup_variant_input_test(self):
        v = variant._setup_variant_input(55);
        self.assertEqual(55.0, v.cpp_variant.getValue(0));

        v = variant._setup_variant_input(variant.linear_interp(points = [(0, 10), (100, 20)]));
        self.assertEqual(15.0, v.cpp_variant.getValue(50));

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
