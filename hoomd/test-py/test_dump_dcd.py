# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os
import tempfile

# unit tests for dump.dcd
class dmp_dcd_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        context.current.sorter.set_params(grid=8)

        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.dcd');
            self.tmp_file = tmp[1]+'.tmp';
        else:
            self.tmp_file = "invalid";

    # tests basic creation of the dump
    def test(self):
        dump.dcd(filename=self.tmp_file, period=100);
        run(100)
        if (comm.get_rank() == 0):
            os.remove(self.tmp_file)

    # tests with phase
    def test_phase(self):
        dump.dcd(filename=self.tmp_file, period=100, phase=0);
        run(100)
        if (comm.get_rank() == 0):
            os.remove(self.tmp_file)

    # tests unwrap_full option
    def test_unwrap_full(self):
        dump.dcd(filename=self.tmp_file, period=100, unwrap_full=True);
        run(100)
        if (comm.get_rank() == 0):
            os.remove(self.tmp_file)

    # tests unwrap_rigid option
    def test_unwrap_rigid(self):
        dump.dcd(filename=self.tmp_file, period=100, unwrap_rigid=True);
        run(100)
        if (comm.get_rank() == 0):
            os.remove(self.tmp_file)

    # tests group option
    def test_group(self):
        typeA = group.type('A');
        dump.dcd(filename=self.tmp_file, group=typeA, period=100);
        run(100)
        if (comm.get_rank() == 0):
            os.remove(self.tmp_file)

    # tests variable periods
    def test_variable(self):
        dump.dcd(filename=self.tmp_file, period=lambda n: n*100);
        run(100)
        if (comm.get_rank() == 0):
            os.remove(self.tmp_file)

    # test disable/enable
    def test_enable_disable(self):
        dcd = dump.dcd(filename=self.tmp_file, period=100);
        dcd.disable()
        self.assertRaises(RuntimeError, dcd.enable)

    # test set_period
    def test_set_period(self):
        dcd = dump.dcd(filename=self.tmp_file, period=100);
        self.assertRaises(RuntimeError, dcd.set_period, 10)

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
