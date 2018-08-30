from __future__ import print_function
from __future__ import division
from hoomd import *
from hoomd import hpmc
import math
import unittest

context.initialize()

class test_ghost_layer(unittest.TestCase):
    def test_implicit(self):
        # setup the MC integration
        system = init.read_snapshot(data.make_snapshot(N=2,box=data.boxdim(Lx=100,Ly=50,Lz=50),particle_types=['A','B']))

        mc = hpmc.integrate.convex_polyhedron(seed=123,implicit=True,depletant_mode='overlap_regions')
        mc.set_params(d=0,a=0)

        mc.set_params(nR=0,depletant_type='B')

        cube_verts=[(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
        mc.shape_param.set('A', vertices=cube_verts)
        mc.shape_param.set('B', vertices=cube_verts)

        system.particles[0].position = (-.2,0,0)
        system.particles[1].position = (1.2,2,0)

        # currently we need this to communicate properly
        run(1)
        self.assertTrue(mc.count_overlaps())

    def test_base(self):
        # setup the MC integration
        system = init.read_snapshot(data.make_snapshot(N=2,box=data.boxdim(Lx=100,Ly=50,Lz=50),particle_types=['A']))

        mc = hpmc.integrate.convex_polyhedron(seed=123)
        mc.set_params(d=0,a=0)

        cube_verts=[(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)]
        mc.shape_param.set('A', vertices=cube_verts)
        self.assertRaises(RuntimeError, mc.shape_param.set, types='B', vertices=cube_verts) #This is an error now

        system.particles[0].position = (-.2,0,0)
        system.particles[1].position = (1.2,2,0)

        # currently we need this to communicate properly
        run(1)
        self.assertTrue(mc.count_overlaps())


    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
