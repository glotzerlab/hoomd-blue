# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
print("Initialized simulation context")
import unittest
import os

# unit tests for options
class option_tests (unittest.TestCase):
    def setUp(self):
        print

    # tests that mode settings work properly
    def test_notice_level(self):
        print("ENTERED TEST_NOTICE_LEVEL")
        print(hoomd.context.current.__dict__)
        hoomd.context.current.device.notice_level = 1;
        self.assertTrue(hoomd.context.current.device.notice_level == 1);
        print("TESTED NOTICE_LEVEL=1")
        hoomd.context.current.device.notice_level = 10
        self.assertTrue(hoomd.context.current.device.notice_level == 10);
        print("TESTED NOTICE_LEVEL=10")

    def tearDown(self):
        pass;


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
