# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os
import gc

# unit tests for init.reset
class init_reset_tests (unittest.TestCase):
    def setUp(self):
        print

    # tests basic creation of the random initializer
    def test_works(self):
        init.create_random(N=100, phi_p=0.05);
        init.reset()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

