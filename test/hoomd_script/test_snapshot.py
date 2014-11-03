# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.create_random
class init_create_snapshot (unittest.TestCase):
    def setUp(self):
        polymer1 = dict(bond_len=1.2, type=['A']*2 + ['B']*3, bond="linear", count=100);
        polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        polymers = [polymer1, polymer2]
        box = data.boxdim(L=35);
        separation=dict(A=0.42, B=0.42)
        self.s = init.create_random_polymers(box=box, polymers=polymers, separation=separation);
        self.assertTrue(self.s);
        self.assertTrue(self.s.sysdef);

    # test taking a snapshot and re-initializing
    def test(self):
        snapshot = self.s.take_snapshot(all=True)
        self.s.restore_snapshot(snapshot)

    # tests options to take_snapshot
    def test_options(self):
        snapshot = self.s.take_snapshot(particles=True)
        self.s.restore_snapshot(snapshot)
        snapshot = self.s.take_snapshot(bonds=True)
        snapshot = self.s.take_snapshot(angles=True)
        snapshot = self.s.take_snapshot(dihedrals=True)
        snapshot = self.s.take_snapshot(impropers=True)
        snapshot = self.s.take_snapshot(walls=True)
        snapshot = self.s.take_snapshot(rigid_bodies=True)
        snapshot = self.s.take_snapshot(integrators=True)

    def test_read_snapshot(self):
        snapshot = self.s.take_snapshot(all=True)
        del self.s
        init.reset()
        self.s = init.read_snapshot(snapshot)

    # test to add and remove some bonds before taking the snapshot
    def test_add_remove_bonds(self):
        l = len(self.s.bonds)
        del(self.s.bonds[l-1])
        del(self.s.bonds[2])
        bonds = []
        for b in self.s.bonds:
            bonds.append((b.a,b.b,b.type))
        self.assertEqual(len(self.s.bonds),l-2)
        snapshot = self.s.take_snapshot(bonds=True)
        self.s.restore_snapshot(snapshot)
        self.assertEqual(len(self.s.bonds),l-2)
        for (b,old_b) in zip(self.s.bonds,bonds):
            new_b = (b.a, b.b, b.type)
            self.assertEqual(new_b,old_b)
        pass

    def tearDown(self):
        del self.s
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
