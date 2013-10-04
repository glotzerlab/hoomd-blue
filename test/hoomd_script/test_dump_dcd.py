# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for dump.dcd
class dmp_dcd_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        dump.dcd(filename="dump_dcd", period=100);
        run(100)
        if (comm.get_rank() == 0):
            os.remove('dump_dcd')

    # tests unwrap_full option
    def test_unwrap_full(self):
        dump.dcd(filename="dump_dcd", period=100, unwrap_full=True);
        run(100)
        if (comm.get_rank() == 0):
            os.remove('dump_dcd')

    # tests unwrap_rigid option
    def test_unwrap_rigid(self):
        # only supported in single-processor mode
        if comm.get_num_ranks()==1:
            dump.dcd(filename="dump_dcd", period=100, unwrap_rigid=True);
            run(100)
            if (comm.get_rank() == 0):
                os.remove('dump_dcd')

    # tests group option
    def test_group(self):
        typeA = group.type('A');
        dump.dcd(filename="dump_dcd", group=typeA, period=100);
        run(100)
        if (comm.get_rank() == 0):
            os.remove('dump_dcd')

    # tests variable periods
    def test_variable(self):
        dump.dcd(filename="dump_dcd", period=lambda n: n*100);
        run(100)
        if (comm.get_rank() == 0):
            os.remove('dump_dcd')

    # test disable/enable
    def test_enable_disable(self):
        dcd = dump.dcd(filename="dump_dcd", period=100);
        dcd.disable()
        self.assertRaises(RuntimeError, dcd.enable)

    # test set_period
    def test_set_period(self):
        dcd = dump.dcd(filename="dump_dcd", period=100);
        self.assertRaises(RuntimeError, dcd.set_period, 10)

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
