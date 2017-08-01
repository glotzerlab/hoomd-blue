from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import os
import numpy

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# test overlaps of faceted sphere configurations
class faceted_sphere(unittest.TestCase):
    def test_special_case(self):
        d=1.0
        offsets = [-0.4, -0.4, -0.4, -0.4]
        normals =[(-0.8660254037844387,-0.49999999999999994,0),
            (-0.8660254037844387,0.49999999999999994,0),
            (0.8660254037844387,0.49999999999999994,0),
            (0.8660254037844387,-0.49999999999999994,0)]
        vertices = [(0.1732050807568878,0.5,0.5),
                (0.4618802153517006,0.0,0.5000000000000001),
                (-0.4618802153517006,0.0,0.5000000000000001),
                (-0.1732050807568878,0.5,0.5),
                (-0.1732050807568878,-0.5,0.5),
                (0.1732050807568878,-0.5,0.5),
                (0.1732050807568878,0.5,-0.5),
                (0.4618802153517005,0.0,-0.5),
                (-0.1732050807568878,0.5,-0.5),
                (-0.4618802153517005,0.0,-0.5),
                (-0.1732050807568878,-0.5,-0.5),
                (0.1732050807568878,-0.5,-0.5)]

        system = create_empty(N=2, box=data.boxdim(Lx=10,Ly=10,Lz=1, dimensions=2), particle_types=['A'])

        system.particles[0].position = (2.02283551,-0.50685786, 0)
        system.particles[1].position = (2.28794172,0.0510459652,0)
        system.particles[0].orientation = (-0.43712055,0,0,-0.89940293)
        system.particles[1].orientation = (-0.86778071,0,0,0.496947323)
        # decrease initialization time with smaller grid for Hilbert curve
        context.current.sorter.set_params(grid=8)

        mc = hpmc.integrate.faceted_sphere(seed=123,d=0,a=0)
        mc.shape_param.set('A',diameter=d, offsets=offsets, vertices=vertices, normals=normals,origin=(0,0,0))

        run(1)
        self.assertEqual(mc.count_overlaps(), 1);

        del mc
        del system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
