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

    # test that adding and removing bonds works with take/restore snapshot
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
        self.s.bonds.add('polymer',0, 10)
        l_new = len(self.s.bonds)
        self.assertEqual(l_new,l-1)
        snapshot = self.s.take_snapshot(bonds=True)
        self.s.restore_snapshot(snapshot)
        l_new = len(self.s.bonds)
        self.assertEqual(l_new,l-1)
        self.assertEqual(self.s.bonds[l_new-1].a,0)
        self.assertEqual(self.s.bonds[l_new-1].b,10)
        self.assertEqual(self.s.bonds[l_new-1].type,'polymer')

    # test removing and adding particles before taking the snapshot
    def test_add_remove_particle(self):
        l = len(self.s.particles)
        l_bonds = len(self.s.bonds)
        tags = []
        # remove the bonds that connect to the particle
        for b in self.s.bonds:
            if b.a == 2 or b.b == 2:
                tags.append(b.tag)

        for t in tags:
            self.s.bonds.remove(t)
        # we should have removed two bonds
        self.assertEqual(len(self.s.bonds),l_bonds-2)

        # remove particle
        del(self.s.particles[2])
        l_new = len(self.s.particles)
        self.assertEqual(l_new,l-1)

        # add particles
        t1 = self.s.particles.add('A')
        t2 = self.s.particles.add('B')
        l_new = len(self.s.particles)
        self.assertEqual(l_new, l+1)
        self.assertEqual(self.s.particles.get(t1).type,'A')
        self.assertEqual(self.s.particles.get(t2).type,'B')
        snapshot = self.s.take_snapshot(all=True)
        self.s.restore_snapshot(snapshot)
        self.assertEqual(len(self.s.particles), l_new)

    def tearDown(self):
        del self.s
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
