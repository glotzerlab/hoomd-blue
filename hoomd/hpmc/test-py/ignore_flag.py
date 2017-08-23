
from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import hoomd
import unittest
import os
import numpy

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# This test tests all possible combinations of the ignore_statistics and ignore_overlaps
# flag for the convex polyhedron class. The ignore flags for all classes are identical
# but the most thorough test would run for all integrators.
#
class pair_ignore_overlaps_check(unittest.TestCase):
    def setUp(self) :
        self.system  = create_empty(N=1000, box=data.boxdim(Lx=11,Ly=5.5, Lz=5.5, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polyhedron(seed=10,a=0.1,d=0.1);

        context.current.sorter.set_params(grid=8)

    def test_overlap_flags(self):

        #particle verts
        rverts= numpy.array( [(-2,-1,-1),
                             (-2,1,-1),
                             (-2,-1,1),
                             (-2,1,1),
                             (2,-1,-1),
                             (2,1,-1),
                             (2,-1,1),
                             (2,1,1)])*0.25

        #init particles on a grid
        xgs,ygs = numpy.meshgrid(numpy.linspace(-4.5,4.5,10),0.5*numpy.linspace(-4.5,4.5,10))
        xs=list()
        ys=list()
        zs=list()
        for z in 0.5*numpy.linspace(-4.5,4.5,10):
            xs.append(xgs)
            ys.append(ygs)
            zs.append(z*numpy.ones(xgs.shape))
        xs = numpy.array(xs).ravel()*1.05
        ys = numpy.array(ys).ravel()*1.05
        zs = numpy.array(zs).ravel()*1.05
        for x,y,z,p in zip(xs,ys,zs,self.system.particles):
            p.position=(x,y,z)
            p.orientation=(1.0,0.0,0.0,0.0)
            p.type='A'


        #track stats and particles do not overlap
        self.mc.shape_param.set('A', vertices=rverts,ignore_statistics=False)
        run(100)

        # verify that not all moves are accepted and zero overlaps are registered
        number_of_overlaps = self.mc.count_overlaps();
        self.assertEqual(number_of_overlaps,0)
        #all rots are accepted
        rotate_acceptance_prob = self.mc.get_rotate_acceptance()
        self.assertLess(rotate_acceptance_prob,1.0)
        #all rots are accepted
        translate_acceptance_prob = self.mc.get_translate_acceptance()
        self.assertLess(translate_acceptance_prob,1.0)

        # Particles cannot overlap, but still track stats
        self.mc.shape_param["A"].ignore_statistics=True
        run(100)
        # verify zero overlaps are registered
        number_of_overlaps = self.mc.count_overlaps();
        self.assertEqual(number_of_overlaps,0)
        # no moves tracked, hoomd says 0 accepted
        rotate_acceptance_prob = self.mc.get_rotate_acceptance()
        self.assertEqual(rotate_acceptance_prob,0.0)
        # no moves tracked, hoomd says 0 accepted
        translate_acceptance_prob = self.mc.get_translate_acceptance()
        self.assertEqual(translate_acceptance_prob,0.0)

        #accept every move by ignoring overlaps, track stats
        self.mc.shape_param["A"].ignore_statistics=False
        self.mc.overlap_checks.set('A','A', False)

        run(100)
        #not all rots are accepted
        rotate_acceptance_prob = self.mc.get_rotate_acceptance()
        self.assertEqual(rotate_acceptance_prob,1)
        #not all rots are accepted
        translate_acceptance_prob = self.mc.get_translate_acceptance()
        if hoomd.context.exec_conf.isCUDAEnabled():
            self.assertLess(translate_acceptance_prob,0.95)
            self.assertGreater(translate_acceptance_prob,0.90)
        else:
            self.assertEqual(translate_acceptance_prob,1)
        #renable overlaps for counting
        self.mc.shape_param["A"].ignore_statistics=False
        self.mc.overlap_checks.set('A','A', True)

        run(1)
        # verify that some overlaps are registered
        number_of_overlaps = self.mc.count_overlaps();
        self.assertGreater(number_of_overlaps,0)

        #reset positions and orientations
        for x,y,z,p in zip(xs,ys,zs,self.system.particles):
            p.position=(x,y,z)
            p.orientation=(1.0,0.0,0.0,0.0)
            p.type='A'

        #renable overlaps for counting
        self.mc.shape_param["A"].ignore_statistics=False
        self.mc.overlap_checks.set('A','A', True)

        run(1)
        # verify that init config contains no overlaps
        number_of_overlaps = self.mc.count_overlaps();
        self.assertEqual(number_of_overlaps,0)

        #accept every move and generate overlapping configs while ignoring stats
        self.mc.shape_param["A"].ignore_statistics=True
        self.mc.overlap_checks.set('A','A', False)

        run(100)
        # no moves tracked, hoomd returns 0
        rotate_acceptance_prob = self.mc.get_rotate_acceptance()
        self.assertEqual(rotate_acceptance_prob,0.0)
        # no moves tracked, hoomd says 0 accepted
        translate_acceptance_prob = self.mc.get_translate_acceptance()
        self.assertEqual(translate_acceptance_prob,0.0)
        #renable overlaps for counting
        self.mc.shape_param["A"].ignore_statistics=False
        self.mc.overlap_checks.set('A','A', True)

        run(1)
        # verify that some overlaps are registered
        number_of_overlaps = self.mc.count_overlaps();
        self.assertGreater(number_of_overlaps,0)

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
