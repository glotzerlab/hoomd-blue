# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest

class analyze_callback_tests(unittest.TestCase):

    def setUp(self):
        init.create_random(N=100, phi_p=0.05)
        self.test_index = 0
        self.test_index_2 = 0

    def test_constructor(self):
        def my_callback(timestep):
            return
        analyze.callback(my_callback, period=1)

    def test_period(self):
        def my_callback(timestep):
            self.test_index += 1
        analyze.callback(callback=my_callback, period=10)
        run(100);
        self.assertEqual(self.test_index, 10)

    def test_phase(self):
        def my_callback(timestep):
            self.test_index += 1

        def out_of_phase_cb(timestep):
            self.test_index_2 += 1

        analyze.callback(callback=my_callback, period=10, phase=10)
        analyze.callback(callback=out_of_phase_cb, period = 10)
        run(9)
        run(89)
        self.assertEqual(self.test_index, 9)
        self.assertEqual(self.test_index_2, 10)

    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
