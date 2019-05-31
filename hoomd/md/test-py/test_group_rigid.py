# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

import hoomd
from hoomd import md
hoomd.context.initialize()
import unittest

class body_group_tests(unittest.TestCase):
    """Tests of groups relating to bodies"""
    def test_rigid(self):
        with hoomd.context.SimulationContext():
            snap = hoomd.data.make_snapshot(N=2, box=hoomd.data.boxdim(L=5), particle_types=['A'])
            if hoomd.comm.get_rank() == 0:
                snap.particles.position[:] = [[1, 0, 0], [2, 0, 0]]

            system = hoomd.init.read_snapshot(snap)

            system.particles.types.add('B')
            rigid = md.constrain.rigid()
            rigid.set_param('A',
                            types=['B']*1,
                            positions=[(0,1,0)]);
            rigid.create_bodies()

            self.assertEqual(len(hoomd.group.rigid_center()), 2)
            self.assertEqual(len(hoomd.group.rigid()), 4)

    def test_floppy(self):
        snap = hoomd.data.make_snapshot(N=8, box=hoomd.data.boxdim(L=2), particle_types=['A'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [[x/10.0, 0, 0] for x in range(8)]
            snap.particles.body[:] = [-2]*4 + [-1]*4

        system = hoomd.init.read_snapshot(snap)
        self.assertEqual(len(hoomd.group.floppy()), 4)

    def test_mixed(self):
        snap = hoomd.data.make_snapshot(N=9, box=hoomd.data.boxdim(L=5), particle_types=['A', 'B'])
        if hoomd.comm.get_rank() == 0:
            snap.particles.typeid[:] = [0]*4 + [1]*5
            snap.particles.position[:] = [[x/10.0, 0, 0] for x in range(9)]

        system = hoomd.init.read_snapshot(snap)

        system.particles.types.add('C')
        rigid = md.constrain.rigid()
        rigid.set_param('A',
                        types=['C']*1,
                        positions=[(0,0.1,0)]);
        rigid.create_bodies()

        for p in system.particles:
            if p.type == 'B':
                p.body = -2

        self.assertEqual(len(hoomd.group.rigid_center()), 4)
        self.assertEqual(len(hoomd.group.rigid()), 8)
        self.assertEqual(len(hoomd.group.nonrigid()), 5)
        self.assertEqual(len(hoomd.group.floppy()), 5)
        self.assertEqual(len(hoomd.group.nonfloppy()), 8)
        self.assertEqual(len(hoomd.group.nonbody()), 0)

    def tearDown(self):
        hoomd.context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
