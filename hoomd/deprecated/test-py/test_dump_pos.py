# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
context.initialize()
import unittest
import os
import tempfile

# unit tests for deprecated.dump.pos
class dmp_pos_tests (unittest.TestCase):
    def setUp(self):
        print
        deprecated.init.create_random(N=100, phi_p=0.05);

        context.current.sorter.set_params(grid=8)

        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.pos');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    # tests basic creation of the dump
    def test(self):
        deprecated.dump.pos(filename=self.tmp_file, period=100);
        run(10)
        if comm.get_rank() == 0:
            os.remove(self.tmp_file)

    # test with phase
    def test_phase(self):
        deprecated.dump.pos(filename=self.tmp_file, period=100, phase=0);
        run(10)
        if comm.get_rank() == 0:
            os.remove(self.tmp_file)

    # tests variable periods
    def test_variable(self):
        deprecated.dump.pos(filename=self.tmp_file, period=lambda n: n*100);
        run(10);
        if comm.get_rank() == 0:
            os.remove(self.tmp_file)

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
