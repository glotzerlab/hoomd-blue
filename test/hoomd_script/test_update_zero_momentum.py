# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests for update.zero_momentum
class update_zero_momentum_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the updater
    def test(self):
        update.zero_momentum()
        run(100);
    
    # test variable periods
    def test_variable(self):
        update.zero_momentum(period = lambda n: n*100);
        run(100);
    
    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

