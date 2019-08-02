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
        self.nl = md.nlist.tree()

    # tests basic creation of the updater
    def test(self):
        dybond = md.update.dynamic_bond(group=group.all(), nlist=self.nl, seed=1, period=1)
        # run(10)

    # def test_formation(self):
    #     dybond = md.update.dynamic_bond(group.all(), nlist=self.nlist, seed=1, period=1)
    #     dybond.set_params(r_cut=2.0, bond_type='harmonic', prob_form=1, prob_break=0)

    # def test_breakage(self):
    #     dybond = md.update.dynamic_bond(group.all(), nlist=self.nlist, seed=1, period=1)
    #     dybond.set_params(r_cut=2.0, bond_type='harmonic', prob_form=0, prob_break=1)

    # def test_cutoff(self)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])