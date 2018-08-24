from __future__ import print_function
from __future__ import division
from hoomd import *
from hoomd import hpmc
import math
import unittest

context.initialize()

class test_clusters_spheres (unittest.TestCase):
    def setUp(self):
        # setup the MC integration
        self.system = init.create_lattice(lattice.sc(a=1.3782337338022654),n=[5,5,5]) #target a packing fraction of 0.2
        self.mc = hpmc.integrate.sphere(seed=123)

        self.mc.shape_param.set('A', diameter=1.0)
        self.clusters = hpmc.update.clusters(self.mc, seed=54321, period=1)

    def test_set_params(self):
        self.clusters.set_params(move_ratio=0.2)
        self.clusters.set_params(flip_probability=0.8)

    def test_integrate(self):
        run(100)

        if comm.get_num_ranks() == 1:
            self.assertAlmostEqual(self.clusters.get_pivot_acceptance(),1.0)
        else:
            # in MPI, there are inactive boundaries
            self.assertTrue(self.clusters.get_pivot_acceptance() > 0)

        self.assertTrue(self.clusters.get_reflection_acceptance() > 0)

    def test_binary_spheres(self):
        self.system.particles.types.add('B')
        self.mc.shape_param.set('B',diameter=1.0)
        self.clusters.set_params(swap_types=['A','B'], swap_move_ratio=0.5, delta_mu=0)
        run(100)

        g = group.type(type='B',name='B')
        self.assertTrue(len(g) > 0)
        if comm.get_num_ranks() == 1:
            self.assertAlmostEqual(self.clusters.get_swap_acceptance(),1.0)
        else:
            self.assertTrue(self.clusters.get_swap_acceptance() > 0)

        # set a finite chemical potential difference
        self.clusters.set_params(delta_mu=0.1)
        run(100)
        self.assertTrue(self.clusters.get_swap_acceptance()<1.0)

    def tearDown(self):
        del self.clusters
        del self.mc
        del self.system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
