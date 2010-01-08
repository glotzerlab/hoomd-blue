# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for dump.mol2
class dmp_mol2_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the dump
    def test(self):
        dump.mol2(filename="dump_mol2", period=100);
        run(101)
        os.remove("dump_mol2.0000000000.mol2")
    
    # tests variable periods
    def test_variable(self):
        dump.mol2(filename="dump_mol2", period=lambda n: n*100);
        run(100);
        os.remove("dump_mol2.0000000000.mol2")
    
    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
