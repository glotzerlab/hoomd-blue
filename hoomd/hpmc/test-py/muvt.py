from hoomd import *
from hoomd import hpmc

import unittest

import math

# this script needs to be run on two ranks

# initialize with one rank per partitions
context.initialize()

class muvt_updater_test(unittest.TestCase):
    def setUp(self):
        self.system = init.create_lattice(lattice.sc(a=8.059959770082347),n=[10,10,10]);

    def tearDown(self):
        del self.muvt
        del self.mc
        del self.system
        context.initialize()

    def test_spheres(self):
        self.mc = hpmc.integrate.sphere(seed=123)
        self.mc.set_params(deterministic=True)
        self.mc.set_params(d=0.1)

        self.mc.shape_param.set('A', diameter=1.0)

        self.muvt=hpmc.update.muvt(mc=self.mc,seed=456,transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_convex_polyhedron(self):
        self.mc = hpmc.integrate.convex_polyhedron(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set("A", vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        self.muvt=hpmc.update.muvt(mc=self.mc,seed=456,transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_sphere_union(self):
        self.mc = hpmc.integrate.sphere_union(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set("A", diameters=[1.0, 1.0], centers=[(-0.25, 0, 0), (0.25, 0, 0)]);

        self.muvt=hpmc.update.muvt(mc=self.mc,seed=456,transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_polyhedron(self):
        self.mc = hpmc.integrate.polyhedron(seed=10);
        self.mc.set_params(deterministic=True)
        import math
        v = [(-0.5, -0.5, 0), (-0.5, 0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (0,0, 1.0/math.sqrt(2)),(0,0,-1.0/math.sqrt(2))];
        f = [(0,4,1),(1,4,2),(2,4,3),(3,4,0),(0,5,1),(1,5,2),(2,5,3),(3,5,0)]
        self.mc.shape_param.set('A', vertices=v, faces=f)

        self.muvt=hpmc.update.muvt(mc=self.mc,seed=456,transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_faceted_sphere(self):
        self.mc = hpmc.integrate.faceted_sphere(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set("A", normals=[(-1,0,0),
                                              (1,0,0),
                                              (0,1,0,),
                                              (0,-1,0),
                                              (0,0,1),
                                              (0,0,-1)],
                                    offsets=[-1]*6,
                                    vertices=[(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)],
                                    diameter=2,
                                    origin=(0,0,0));

        self.muvt=hpmc.update.muvt(mc=self.mc,seed=456,transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_spheropolyhedron(self):
        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set("A", vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        self.muvt=hpmc.update.muvt(mc=self.mc,seed=456,transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_ellipsoid(self):
        self.mc = hpmc.integrate.ellipsoid(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);

        self.muvt=hpmc.update.muvt(mc=self.mc,seed=456,transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)


class muvt_updater_test_2d(unittest.TestCase):
    def setUp(self):
        self.system = init.create_lattice(lattice.sq(a=8.059959770082347),n=[10,10]);

    def tearDown(self):
        del self.muvt
        del self.mc
        del self.system
        context.initialize()

    def test_spheres(self):
        self.mc = hpmc.integrate.sphere(seed=0)
        self.mc.set_params(deterministic=True)
        self.mc.set_params(d=0.1)
        self.mc.shape_param.set('A', diameter=1.0)

        self.muvt=hpmc.update.muvt(mc=self.mc, seed=456, transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_convex_polygon(self):
        self.mc = hpmc.integrate.convex_polygon(seed=0)
        self.mc.set_params(deterministic=True)
        self.mc.set_params(d=0.1)

        square_vertices = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        self.mc.shape_param.set('A', vertices=square_vertices)

        self.muvt=hpmc.update.muvt(mc=self.mc, seed=456, transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

    def test_simple_polygon(self):
        self.mc = hpmc.integrate.convex_polygon(seed=0)
        self.mc.set_params(deterministic=True)
        self.mc.set_params(d=0.1)

        vertices = [[-0.5, -0.5], [-0.25, -0.5], [-0.25, -0.25], [0.25, -0.25],
                    [0.25, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        self.mc.shape_param.set('A', vertices=vertices)

        self.muvt=hpmc.update.muvt(mc=self.mc, seed=456, transfer_types=['A'])
        self.muvt.set_fugacity('A', 100)

        run(100)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
