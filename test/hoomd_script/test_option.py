# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random
class init_create_random_tests (unittest.TestCase):
    def setUp(self):
        print
    
    # tests that mode settings work properly
    def test_mode(self):
        option.set_mode('gpu');
        self.assert_(globals.options.mode == 'gpu');

        option.set_mode('cpu');
        self.assert_(globals.options.mode == 'cpu');

        option.set_mode(None);
        self.assert_(globals.options.mode is None);
        
        self.assertRaises(RuntimeError, option.set_mode, 'foo');
    
    def test_gpu(self):
        option.set_gpu(1);
        self.assert_(globals.options.gpu == 1);
        self.assert_(globals.options.mode == 'gpu');

        option.set_gpu(None);
        self.assert_(globals.options.gpu is None);

        self.assertRaises(RuntimeError, option.set_gpu, 'foo');

    def test_ncpu(self):
        option.set_ncpu(1);
        self.assert_(globals.options.ncpu == 1);
        self.assert_(globals.options.mode == 'cpu');

        option.set_ncpu(None);
        self.assert_(globals.options.ncpu is None);

        self.assertRaises(RuntimeError, option.set_ncpu, 'foo');

    def test_gpu_error_checking(self):
        option.set_gpu_error_checking(False);
        self.assert_(globals.options.gpu_error_checking == False);

        option.set_gpu_error_checking(True);
        self.assert_(globals.options.gpu_error_checking == True);

    def test_min_cpu(self):
        option.set_min_cpu(False);
        self.assert_(globals.options.min_cpu == False);

        option.set_min_cpu(True);
        self.assert_(globals.options.min_cpu == True);

    def test_ignore_display(self):
        option.set_ignore_display(False);
        self.assert_(globals.options.ignore_display == False);

        option.set_ignore_display(True);
        self.assert_(globals.options.ignore_display == True);
        
    def tearDown(self):
        pass;


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

