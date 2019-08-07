# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os

# unit tests for options
class option_tests (unittest.TestCase):
    def setUp(self):
        print

    # tests that mode settings work properly
    def test_notice_level(self):
        hoomd.context.current.device.notice_level = 1;
        self.assertTrue(hoomd.context.current.device.notice_level == 1);

        hoomd.context.current.device.notice_level = 10
        self.assertTrue(hoomd.context.current.device.notice_level == 10);

    def tearDown(self):
        pass;


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
