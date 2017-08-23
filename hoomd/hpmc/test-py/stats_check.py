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

# This test ensures that the small box code path is enabled at the correct box sizes and works correctly
# It performs two tests
# 1) Initialize a system with known overlaps (or not) and verify that count_overlaps produces the correct result
# 2) Run many steps tracking overlap counts and trial moves accepted
#
# Success condition: Correctly functioning code should enable the small box code path and report overlaps when they
# are created and none during the run(). Some moves should be accepted and some should be rejected.
#
# Failure mode 1: If the box size is between 1 and 2 diameters and the cell list code path activates, it is an error
#
# Failure mode 2: If the small box trial move code path does not correctly check the updated orientation when checking
# particles vs its own image - some number of overlaps will show up during the run().
#
# To detect these failure modes, a carefully designed system is needed. Place a cube (side length 1) in a cubic box
# 1 < L < sqrt(2). This allows the square to rotate to many possible orientations - but some are disallowed.
# For example, the corners of the square will overlap at 45 degrees.

#when all particles ignored, make sure no attempted moves are registered by the counter
class pair_accept_all (unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1000, box=data.boxdim(L=12, dimensions=3), particle_types=['A'])

        self.mc = hpmc.integrate.ellipsoid(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set('A', a=0.5,b=0.25,c=0.15,ignore_statistics=True)
        self.mc.overlap_checks.set('A','A', False)

        context.current.sorter.set_params(grid=8)

    def test_accept_all(self):
        # check 1, see if there are any overlaps. There should be none as all overlaps are ignored

        #init particles on a grid
        xgs,ygs = numpy.meshgrid(numpy.linspace(-4.5,4.5,10),numpy.linspace(-4.5,4.5,10))
        xs=list()
        ys=list()
        zs=list()
        for z in numpy.linspace(-4.5,4.5,10):
            xs.append(xgs)
            ys.append(ygs)
            zs.append(z*numpy.ones(xgs.shape))
        xs = numpy.array(xs).ravel()
        ys = numpy.array(ys).ravel()
        zs = numpy.array(zs).ravel()

        for x,y,z,p in zip(xs,ys,zs,self.system.particles):
            p.position=(x,y,z)
            p.orientation=(1.0,0.0,0.0,0.0)

        del p
        run(100)


        # verify that all moves are accepted and zero overlaps are registered
        number_of_overlaps = self.mc.count_overlaps();
        self.assertEqual(number_of_overlaps,0)

        #when all particles are ignored the acceptance probs are set to zero
        #to avoid a div by zero error
        translate_acceptance_prob = self.mc.get_translate_acceptance()
        self.assertEqual(translate_acceptance_prob,0)

        rotate_acceptance_prob = self.mc.get_rotate_acceptance()
        self.assertEqual(rotate_acceptance_prob,0)

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

#when none of the particles are ignored, make sure that moves that generate overlapping configs are rejected
class pair_accept_none (unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=2, box=data.boxdim(L=10, dimensions=3), particle_types=['A'])

        self.mc = hpmc.integrate.ellipsoid(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set('A', a=0.5,b=0.25,c=0.15,ignore_statistics=False)
        self.mc.set_params(d=0.01,a=0.01)
        context.current.sorter.set_params(grid=8)

    def test_accept_none(self):
        # check 1, see if there are any overlaps. There should be 2

        #init particles on a grid
        for p in self.system.particles:
            p.position=(0.0,0.0,0.0)
            p.orientation=(1.0,0.0,0.0,0.0)

        del p
        run(100)

        # verify no moves are accepted and overlaps are registered
        number_of_overlaps = self.mc.count_overlaps();
        self.assertEqual(number_of_overlaps,1)

        translate_acceptance_prob = self.mc.get_translate_acceptance()
        self.assertEqual(translate_acceptance_prob,0)

        rotate_acceptance_prob = self.mc.get_rotate_acceptance()
        self.assertEqual(rotate_acceptance_prob,0)

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

#ignore a fraction of the particles, and make sure the acceptance probabilities are only counted for the not
#ignored particles.
class pair_accept_some(unittest.TestCase):
    def setUp(self) :
        self.system  = create_empty(N=1000, box=data.boxdim(L=12, dimensions=3), particle_types=['A','B'])

        self.mc = hpmc.integrate.ellipsoid(seed=84,d=1.0);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set('A', a=0.5,b=0.5,c=0.5,ignore_statistics=True)
        self.mc.overlap_checks.set('A','A', False)
        self.mc.shape_param.set('B', a=0.5,b=0.5,c=0.5,ignore_statistics=False)

        context.current.sorter.set_params(grid=8)

    def test_accept_some(self):
        # check 1, see if there are any overlaps. There should be none as all overlaps are ignored

        #init particles on a grid
        xgs,ygs = numpy.meshgrid(numpy.linspace(-4.5,4.5,10),numpy.linspace(-4.5,4.5,10))
        xs=list()
        ys=list()
        zs=list()
        for z in numpy.linspace(-4.5,4.5,10):
            xs.append(xgs)
            ys.append(ygs)
            zs.append(z*numpy.ones(xgs.shape))
        xs = numpy.array(xs).ravel()*1.05
        ys = numpy.array(ys).ravel()*1.05
        zs = numpy.array(zs).ravel()*1.05
        for x,y,z,p in zip(xs,ys,zs,self.system.particles):
            p.position=(x,y,z)
            p.orientation=(1.0,0.0,0.0,0.0)

        #loop over ignored particle fractions, and expected accepted probs for the 'B' particles
        gpu_accept_probs = [0.023,0.023,0.025,0.075]
        cpu_accept_probs = [0.06,0.06,0.07,0.29]
        if  context.exec_conf.isCUDAEnabled():
            probs = gpu_accept_probs
        else:
            probs = cpu_accept_probs
        for N_a,prob in zip([0,10,100,999],probs):
            for t,p in zip(['A']*N_a+['B']*(1000-N_a),self.system.particles):
                p.type=t

            del p
            run(100)

            # verify that all moves are accepted and zero overlaps are registered
            number_of_overlaps = self.mc.count_overlaps();
            self.assertEqual(number_of_overlaps,0)

            #assert the the acceptance prob is within acceptable bounds
            translate_acceptance_prob = self.mc.get_translate_acceptance()
            self.assertGreater(translate_acceptance_prob,prob*0.5)
            self.assertLess(translate_acceptance_prob,prob*2.0)

            #should be zero, because these are spheres and no rotation moves should be attempted
            rotate_acceptance_prob = self.mc.get_rotate_acceptance()
            self.assertEqual(rotate_acceptance_prob,0)

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
