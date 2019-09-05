# -*- coding: iso-8859-1 -*-

from hoomd import *
from hoomd import md

context.initialize()
import unittest
import os

r_colloid = 1
center_from_origin = 1
# tests for md.update.dynamic_bond
class update_dynamic_bond(unittest.TestCase):
    def setUp(self):
        print
        snapshot = data.make_snapshot(N=2, box=data.boxdim(Lx=80, Ly=80, Lz=80),
                                      bond_types=['polymer'], particle_types=['colloid'])
        snapshot.particles.diameter[:] = [r_colloid*2]*2
        snapshot.particles.position[0] = [-center_from_origin, 0, 0]
        snapshot.particles.position[1] = [center_from_origin, 0, 0]
        self.system = init.read_snapshot(snapshot)

    # tests basic creation of the updater
    def test(self):
        nl = md.nlist.tree()
        dybond = md.update.dynamic_bond(group=group.all(), nlist=nl, seed=1, period=1)
        # run(10)


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])