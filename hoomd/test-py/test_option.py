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
        option.set_notice_level(1);
        self.assert_(hoomd.context.options.notice_level == 1);

        option.set_notice_level(10);
        self.assert_(hoomd.context.options.notice_level == 10);

        self.assertRaises(RuntimeError, option.set_notice_level, 'foo');

    def tearDown(self):
        pass;


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
