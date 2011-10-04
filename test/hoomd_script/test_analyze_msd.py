# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for analyze.msd
class analyze_msd_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # tests basic creation of the analyzer
    def test(self):
        analyze.msd(period = 10, filename="test.log", groups=[group.all()]);
        run(100);

    # test variable period
    def test_variable(self):
        analyze.msd(period = lambda n: n*10, filename="test.log", groups=[group.all()]);
        run(100);
    
    # test error if no groups defined
    def test_no_gropus(self):
        self.assertRaises(RuntimeError, analyze.msd, period=10, filename="test.log", groups=[]);
    
    # test set_params
    def test_set_params(self):
        ana = analyze.msd(period = 10, filename="test.log", groups=[group.all()]);
        ana.set_params(delimiter = ' ');
        run(100);
    
    def tearDown(self):
        init.reset();
        os.remove("test.log");


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

