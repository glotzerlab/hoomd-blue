# -*- coding: iso-8859-1 -*-
# Maintainer: jglaser

from hoomd_script import *
init.setup_exec_conf();
import unittest
import os

# unit test to run a simple polymer system with pair and bond potentials
class replicate(unittest.TestCase):
    def test_barrier(self):
        comm.barrier();

    def test_barrier_all(self):
        comm.barrier_all();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
