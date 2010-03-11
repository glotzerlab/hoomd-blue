# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for analyze.log
class analyze_log_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # tests basic creation of the analyzer
    def test(self):
        analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename="test.log");
        run(100);
    
    # test set_params
    def test_set_params(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename="test.log");
        ana.set_params(quantities = ['test1']);
        run(100);
        ana.set_params(delimiter = ' ');
        run(100);
        ana.set_params(quantities = ['test2', 'test3'], delimiter=',')
        run(100);

    # test variable period
    def test_variable(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = lambda n: n*10, filename="test.log");
        run(100);        
    
    # test the initialization checks
    def test_init_checks(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename="test.log");
        ana.cpp_analyzer = None;
        
        self.assertRaises(RuntimeError, ana.enable);
        self.assertRaises(RuntimeError, ana.disable);
    
    def tearDown(self):
        init.reset();
        os.remove("test.log");


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

