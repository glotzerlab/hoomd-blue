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

# Tests to ensure that all particle types can be created

class validate_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=2, box=data.boxdim(L=10, dimensions=2), particle_types=['A', 'B'])

        self.mc = hpmc.integrate.convex_polygon(seed=10);
        self.mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);

        context.current.sorter.set_params(grid=8)

    def test_sanity_check(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        self.system.particles[1].position = (1.5,1.5,1.5);
        self.system.particles[1].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        self.assertRaises(RuntimeError, run, 1);
        self.mc.shape_param.set('B', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);
        run(1)
        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()


class convex_polygon_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.convex_polygon(seed=10);
        self.mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);

        context.current.sorter.set_params(grid=8)

    def test_convex_polygon(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

class simple_polygon_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.simple_polygon(seed=10);
        self.mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.0, 0.0), (0.5, 0.5), (-0.5, 0.5)]);

        context.current.sorter.set_params(grid=8)

    def test_simple_polygon(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

class convex_polyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.convex_polyhedron(seed=10);
        self.mc.shape_param.set('A', vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        context.current.sorter.set_params(grid=8)

    def test_convex_polyhedron(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

class sphere_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.sphere(seed=10);
        self.mc.shape_param.set('A', diameter=1.0)

        context.current.sorter.set_params(grid=8)

    def test_sphere(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

class sphere_union_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=3), particle_types=['A'])

        self.mc = hpmc.integrate.sphere_union(seed=10,capacity=16);
        self.mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0, 0), (0.25, 0, 0)]);

        context.current.sorter.set_params(grid=8)

    def test_sphere_union(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

class convex_spheropolygon_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.convex_spheropolygon(seed=10);
        self.mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);

        context.current.sorter.set_params(grid=8)

    def test_convex_spheropolygon(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

class polyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.polyhedron(seed=10);
        self.mc.shape_param.set('A', vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
                                (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)],\
                                faces = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)]);

        context.current.sorter.set_params(grid=8)

    def test_polyhedron(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

# apparently not all params are being set here, but I do not see proper documentation
# so it is just commented out until it is updated by the author
class faceted_sphere_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.faceted_sphere(seed=10);
        self.mc.shape_param.set('A', normals=[(-1,0,0),
                                              (1,0,0),
                                              (0,1,0,),
                                              (0,-1,0),
                                              (0,0,1),
                                              (0,0,-1)],
                                    offsets=[-1]*6,
                                    vertices=[(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)],
                                    diameter=2,
                                    origin=(0,0,0));

        context.current.sorter.set_params(grid=8)

    def test_faceted_sphere(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

class convex_spheropolyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=10);
        self.mc.shape_param.set('A', vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        context.current.sorter.set_params(grid=8)

    def test_convex_spheropolyhedron(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

class ellipsoid_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.ellipsoid(seed=10);
        self.mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);

        context.current.sorter.set_params(grid=8)

    def test_ellipsoid(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

class sphinx_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])

        self.mc = hpmc.integrate.sphinx(seed=10);
        self.mc.shape_param.set('A', diameters=[2,-2.2,-2.2], centers=[(0,0,0), (0,0,1.15), (0,0,-1.15)], \
                           colors=['ff','ffff00','ffff00']);

        context.current.sorter.set_params(grid=8)

    def test_sphinx(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # verify that the particle is created correctly
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
