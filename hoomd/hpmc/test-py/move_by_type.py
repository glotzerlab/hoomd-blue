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

# This test ensures that the small box code path is enabled at the correct box sizes and works correctly
# It performs two tests
# 1) we set translations moves by type, freezing all, half then none of the system. We use acceptance probabilities to check that system is behaving as expected.
# 2) we set rotation  moves by type, freezing all, half then none of the system. We use acceptance probabilities to check that system is behaving as expected.
#
# Success condition: Correctly reproduce the precalculated acceptance probabilities for these move sizes
#
# Failure mode: Failing to set the move size by type correctly will change the acceptance prob and break these tests
#
# Failure mode 2: Moving particles for which move size is zero
#
class pair_move_some(unittest.TestCase):
    def setUp(self) :
        self.system  = create_empty(N=1000, box=data.boxdim(Lx=11,Ly=5.5, Lz=5.5, dimensions=3), particle_types=['A','B'])

        self.mc = hpmc.integrate.convex_polyhedron(seed=10,a=0.0,d={'A':0.1,'B':0.0});
        self.mc.set_params(deterministic=True)
        rverts= numpy.array( [(-2,-1,-1),
                             (-2,1,-1),
                             (-2,-1,1),
                             (-2,1,1),
                             (2,-1,-1),
                             (2,1,-1),
                             (2,-1,1),
                             (2,1,1)])*0.25

        self.mc.shape_param.set('A', vertices=rverts,ignore_statistics=False)
        self.mc.shape_param.set('B', vertices=rverts,ignore_statistics=False)

        context.current.sorter.set_params(grid=8)

    def test_move_some(self):

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

        #loop over ignored particle fractions, and expected accepted probs for the 'B' particles
        gpu_accept_probs = [1.0, 0.57,  0.14]
        cpu_accept_probs = [1.0,0.597, 0.147]
        if hoomd.context.exec_conf.isCUDAEnabled():
            probs = gpu_accept_probs
        else:
            probs = cpu_accept_probs
        for N_a,prob in zip([0,500,1000],probs):
            for t,p in zip(['A']*N_a+['B']*(1000-N_a),self.system.particles):
                p.type=t

            run(100)

            #check that the B particles haven't moved
            for x,y,z,p in zip(xs,ys,zs,self.system.particles):
                if(p.type=='B'):
                    r0 = p.position
                    self.assertAlmostEqual(r0[0],x,places=4)
                    self.assertAlmostEqual(r0[1],y,places=4)
                    self.assertAlmostEqual(r0[2],z,places=4)
            del p

            # verify that all moves are accepted and zero overlaps are registered
            number_of_overlaps = self.mc.count_overlaps();
            self.assertEqual(number_of_overlaps,0)

            #all rots are accepted (0 movesize)
            rotate_acceptance_prob = self.mc.get_rotate_acceptance()
            self.assertAlmostEqual(rotate_acceptance_prob,1.0,places=3)

            #some trans are accepted (check prob)
            translate_acceptance_prob = self.mc.get_translate_acceptance()
            self.assertGreater(translate_acceptance_prob,prob*0.9)
            self.assertLess(translate_acceptance_prob,prob*1.1)

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

class pair_rot_some(unittest.TestCase):
    def setUp(self) :
        self.system  = create_empty(N=1000, box=data.boxdim(Lx=11,Ly=5.5, Lz=5.5, dimensions=3), particle_types=['A','B'])

        self.mc = hpmc.integrate.convex_polyhedron(seed=10,d=0.0,a={'A':0.05,'B':0.0});
        self.mc.set_params(deterministic=True)
        rverts= numpy.array( [(-2,-1,-1),
                             (-2,1,-1),
                             (-2,-1,1),
                             (-2,1,1),
                             (2,-1,-1),
                             (2,1,-1),
                             (2,-1,1),
                             (2,1,1)])*0.25

        self.mc.shape_param.set('A', vertices=rverts,ignore_statistics=False)
        self.mc.shape_param.set('B', vertices=rverts,ignore_statistics=False)

        context.current.sorter.set_params(grid=8)

    def test_rot_some(self):

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

        #loop over ignored particle fractions, and expected accepted probs for the 'B' particles
        gpu_accept_probs = [1.0, 0.517, 0.031]
        cpu_accept_probs = [1.0, 0.517, 0.0299]
        if hoomd.context.exec_conf.isCUDAEnabled():
            probs = gpu_accept_probs
        else:
            probs = cpu_accept_probs
        for N_a,prob in zip([0,500,1000],probs):
            for t,p in zip(['A']*N_a+['B']*(1000-N_a),self.system.particles):
                p.type=t

            run(100)

            #check that B orientations are unchanged
            for p in self.system.particles:
                if(p.type=='B'):
                    q0 = p.orientation
                    self.assertAlmostEqual(q0[0],1)
                    self.assertAlmostEqual(q0[1],0)
                    self.assertAlmostEqual(q0[2],0)
                    self.assertAlmostEqual(q0[2],0)
            del p

            # verify that all moves are accepted and zero overlaps are registered
            number_of_overlaps = self.mc.count_overlaps();
            self.assertEqual(number_of_overlaps,0)

            translate_acceptance_prob = self.mc.get_translate_acceptance()
            #sometimes gridshift will cause a very small number of rejections
            self.assertAlmostEqual(translate_acceptance_prob,1.0,places=3)

            #should be zero, because these are spheres and no rotation moves should be attempted
            rotate_acceptance_prob = self.mc.get_rotate_acceptance()
            self.assertGreater(rotate_acceptance_prob,prob*0.7)
            self.assertLess(rotate_acceptance_prob,prob*1.3)

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
