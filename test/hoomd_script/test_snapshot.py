# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random
class init_create_snapshot (unittest.TestCase):
    def setUp(self):
        pass
    
    # tests basic creation of the random initializer
    def test(self):
        system = init.create_random(N=100, phi_p=0.05);
        self.assertTrue(globals.system_definition);
        self.assertTrue(globals.system);
        snapshot = system.take_snapshot()
        init.restore_from_snapshot(snapshot)
        del system
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

