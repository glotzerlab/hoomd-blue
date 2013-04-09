# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random
class init_create_snapshot (unittest.TestCase):
    def setUp(self):
        pass
    
    # test taking a snapshot and re-initializing
    def test(self):
        system = init.create_random(N=100, phi_p=0.05);
        self.assertTrue(globals.system_definition);
        self.assertTrue(globals.system);
        snapshot = system.take_snapshot(all=True)
        init.restore_from_snapshot(snapshot)
        del system

    # tests options to take_snapshot
    def test_options(self):
        system = init.create_random(N=100, phi_p=0.05);
        self.assertTrue(globals.system_definition);
        self.assertTrue(globals.system);
        snapshot = system.take_snapshot(particles=True)
        init.restore_from_snapshot(snapshot)
        snapshot = system.take_snapshot(bonds=True)
        snapshot = system.take_snapshot(angles=True)
        snapshot = system.take_snapshot(dihedrals=True)
        snapshot = system.take_snapshot(impropers=True)
        snapshot = system.take_snapshot(walls=True)
        snapshot = system.take_snapshot(rigid_bodies=True)
        snapshot = system.take_snapshot(integrators=True)
        del system
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

