# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;

from hoomd import _hoomd
if _hoomd.is_TBB_available():
    # test the command line option
    context.initialize("--nthreads=4")

    import unittest
    import os

    # unit tests for options
    class option_tests (unittest.TestCase):
        def setUp(self):
            print

        # tests that mode settings work properly
        def test_nthreads(self):
            self.assertEqual(hoomd.context.ExecutionContext().num_threads, 4);
            option.set_num_threads(2)
            self.assertEqual(hoomd.context.ExecutionContext().num_threads, 2);

        def tearDown(self):
            pass;

    if __name__ == '__main__':
        unittest.main(argv = ['test.py', '-v'])
