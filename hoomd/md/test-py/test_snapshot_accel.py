# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
from hoomd import md
import hoomd;
context.initialize()
import unittest
import os
import sys
import numpy

# unit tests for init.take_snapshot and init.restore_snapshot
class init_snapshot_accel (unittest.TestCase):
    def setUp(self):
        polymer1 = dict(bond_len=1.2, type=['A']*2 + ['B']*3, bond="linear", count=100);
        polymer2 = dict(bond_len=1.2, type=['B']*4, bond="linear", count=10)
        polymers = [polymer1, polymer2]
        box = data.boxdim(L=35);
        separation=dict(A=0.42, B=0.42)
        self.s = deprecated.init.create_random_polymers(box=box, polymers=polymers, separation=separation);
        self.assertTrue(self.s);
        self.assertTrue(self.s.sysdef);

        # add some constraints
        self.s.constraints.add(0, 1, 0.1)
        self.s.constraints.add(0, 2, 0.2)
        self.s.constraints.add(3, 4, 3.4)

    def test_default_accel_set(self):
        snapshot = self.s.take_snapshot(all=True)
        self.assertFalse(snapshot.particles.is_accel_set);

    def test_integrator_sets_accel(self):
        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group=hoomd.group.all());
        run(1)
        snapshot = self.s.take_snapshot(all=True)
        self.assertTrue(snapshot.particles.is_accel_set);

    def test_snapshot_accel_set_passthrough(self):
        snapshot = self.s.take_snapshot(particles=True)
        self.assertFalse(snapshot.particles.is_accel_set);

        md.integrate.mode_standard(dt=0)
        md.integrate.nve(group=hoomd.group.all());
        run(1)
        snapshot2 = self.s.take_snapshot(all=True)
        self.assertTrue(snapshot2.particles.is_accel_set);

        self.s.restore_snapshot(snapshot)
        snapshot3 = self.s.take_snapshot(all=True)
        self.assertFalse(snapshot3.particles.is_accel_set);

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
