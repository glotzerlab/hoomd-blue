from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import os
import numpy
import math

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# test overlaps of faceted ellipsoid configurations
class faceted_ellipsoid(unittest.TestCase):
    def test_half_ellpisoid(self):
        # a prolate ellipsoid with -x halfspace cut
        a = 0.5
        b = 0.5
        c = 4
        offsets = [0]
        normals =[(-1,0,0)]
        vertices = []

        system = create_empty(N=2, box=data.boxdim(Lx=10,Ly=10,Lz=1, dimensions=2), particle_types=['A'])

        # slightly non overlapping
        system.particles[0].position = (0,0,0)
        system.particles[1].position = (-0.001,0,0)
        system.particles[0].orientation = (1,0,0,0)
        system.particles[1].orientation = (0,0,0,1) # rotate around z axis by pi
        # decrease initialization time with smaller grid for Hilbert curve
        context.current.sorter.set_params(grid=8)

        mc = hpmc.integrate.faceted_ellipsoid(seed=123,d=0,a=0)
        mc.shape_param.set('A',a=a, b=b, c=c, offsets=offsets, vertices=vertices, normals=normals,origin=(0,0,0))

        run(1)
        self.assertEqual(mc.count_overlaps(), 0);

        # let them overlap slightly
        system.particles[1].position = (0.001,0,0)
        run(1)
        self.assertEqual(mc.count_overlaps(), 1);

        # slight overlap with facets facing away from each other
        system.particles[1].position = (0.99,0,0)
        run(1)
        self.assertEqual(mc.count_overlaps(), 1);

        # barely not overlapping with facets facing away from each other
        system.particles[1].position = (1.01,0,0)
        run(1)
        self.assertEqual(mc.count_overlaps(), 0);

        del mc
        del system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
