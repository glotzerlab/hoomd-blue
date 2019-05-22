from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import *
from hoomd import hpmc

import unittest
import os
import numpy as np
import numpy
import math

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs)
    return init.read_snapshot(snap)

class sphere_wall_sphere_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.sphere(seed=10)
        self.mc.shape_param.set('A', diameter=1.0)

        self.ext_wall = hpmc.field.wall(self.mc)
        self.ext_wall.add_sphere_wall(5.0, origin=[0,0,0], inside=True)


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall
        self.system.particles[0].position = (0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)

        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)

        # b. inside=False: then there should be an overlap
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

        run(100)

        # 2. test a particle outside the wall
        self.system.particles[0].position = (10,0,0)
        # a. inside=False
        self.assertEqual(self.ext_wall.count_overlaps(), 0)

        run(100)
        self.assertTrue(self.system.particles[0].position != (10,0,0))
        self.system.particles[0].position = (10,0,0)

        # b. inside=True
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=True)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

        run(100)
        p = self.system.particles[0].position;
        numpy.testing.assert_allclose(p[0], 10)
        self.assertTrue(math.fabs(p[1]) < 1e-5)
        self.assertTrue(math.fabs(p[2]) < 1e-5)

    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class sphere_wall_convex_polyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polyhedron(seed=10)
        self.mc.shape_param.set('A', vertices = [(-0.5,-0.5,-0.5),
                                                (-0.5,0.5,-0.5),
                                                (-0.5,-0.5,0.5),
                                                (-0.5,0.5,0.5),
                                                (0.5,-0.5,-0.5),
                                                (0.5,0.5,-0.5),
                                                (0.5,-0.5,0.5),
                                                (0.5,0.5,0.5)])

        self.ext_wall = hpmc.field.wall(self.mc)
        self.ext_wall.add_sphere_wall(5.0, origin=[0,0,0], inside=True)


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall, far from the boundary
        self.system.particles[0].position = (0,0,0)
        self.system.particles[0].orientation = (1,0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)

        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)
        # b. inside=False: then there should be an overlap
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))

        # 2. test a particle just overlapping the wall
        # a. inside=False: this is a pathological case wherein all vertices lie outside the sphere, but
        # the cube still intersects the sphere. count_overlaps should give 1 in this case (ie badly confined).
        # the intersection is between the face of the cube parallel to the yz axis, and the sphere.
        for x in np.linspace(5.499999, 4.5, 1000):
            self.system.particles[0].position = (x,0,0)
            hoomd.util.quiet_status()
            self.assertEqual(self.ext_wall.count_overlaps(), 1)
            hoomd.util.unquiet_status()
        # b. inside=True: this should return an overlap.
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=True)
        for x in np.linspace(4.475,5.475,1000):
            self.system.particles[0].position = (x,0,0)
            hoomd.util.quiet_status()
            self.assertEqual(self.ext_wall.count_overlaps(), 1)
            hoomd.util.unquiet_status()

        # 3. test a particle not quite overlapping the wall
        self.system.particles[0].position = (4.14,0,0)
        # a. inside=True
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside=False
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

        # 4. test a particle outside the wall, far from the boundary
        self.system.particles[0].position = (10,0,0)
        # a. inside = False: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside = True: then there should be an overlap
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=True)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)



    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class sphere_wall_convex_spheropolyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=10)
        self.mc.shape_param.set('A', vertices = [(-0.5,-0.5,-0.5),
                                                (-0.5,0.5,-0.5),
                                                (-0.5,-0.5,0.5),
                                                (-0.5,0.5,0.5),
                                                (0.5,-0.5,-0.5),
                                                (0.5,0.5,-0.5),
                                                (0.5,-0.5,0.5),
                                                (0.5,0.5,0.5)],
                                                sweep_radius=0.5)

        self.ext_wall = hpmc.field.wall(self.mc)
        self.ext_wall.add_sphere_wall(5.0, origin=[0,0,0], inside=True)


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall, far from the boundary
        self.system.particles[0].position = (0,0,0)
        self.system.particles[0].orientation = (1,0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)
        # b. inside=False: then there should be an overlap
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))

        # 2. test a particle just overlapping the wall
        # a. inside=False: this is a pathological case wherein all vertices lie outside the sphere, but
        # the cube still intersects the sphere. count_overlaps should give 1 in this case (ie badly confined).
        # the intersection is between the face of the cube parallel to the yz axis, and the sphere.
        for x in np.linspace(5.999999, 4.0, 1000):
            self.system.particles[0].position = (x,0,0)
            hoomd.util.quiet_status()
            self.assertEqual(self.ext_wall.count_overlaps(), 1)
            hoomd.util.unquiet_status()
        # b. inside=True: this should return an overlap.
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=True)
        for x in np.linspace(4.0,5.999999,1000):
            self.system.particles[0].position = (x,0,0)
            hoomd.util.quiet_status()
            self.assertEqual(self.ext_wall.count_overlaps(), 1)
            hoomd.util.unquiet_status()

    def test_individual(self):
        # 3. test a particle not quite overlapping the wall
        self.system.particles[0].position = (3.64,0,0)
        # a. inside=True
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside=False
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

        # 4. test a particle outside the wall, far from the boundary
        self.system.particles[0].position = (10,0,0)
        # a. inside = False: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside = True: then there should be an overlap
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=True)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)


    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class sphere_wall_convex_spheropolyhedron_sphere_test(unittest.TestCase):
    """Test that pure spheres work for the spheropolyhedron integrator"""
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=10)
        self.mc.shape_param.set('A', vertices = [], sweep_radius=1)

        self.ext_wall = hpmc.field.wall(self.mc)
        self.ext_wall.add_sphere_wall(5.0, origin=[0,0,0], inside=True)


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall, far from the boundary
        self.system.particles[0].position = (0,0,0)
        self.system.particles[0].orientation = (1,0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))

        # Now have the particle overlapping the wall
        self.system.particles[0].position = (4.5,0,0)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        # Need allclose since 4.5 can run into floating point issues
        self.assertTrue(
                np.allclose(
                    self.system.particles[0].position,
                        (4.5,0,0)
                        )
                )

        # b. inside=False: then there should be an overlap
        self.system.particles[0].position = (0,0,0)
        self.ext_wall.set_sphere_wall(0, 5.0, origin=[0,0,0], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))


    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class cylinder_wall_sphere_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.sphere(seed=10)
        self.mc.shape_param.set('A', diameter=1.0)

        self.ext_wall = hpmc.field.wall(self.mc)
        self.ext_wall.add_cylinder_wall(5.0, [0,0,0], [0,0,1], inside=True)


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall
        self.system.particles[0].position = (0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)
        # b. inside=False: then there should be an overlap
        self.ext_wall.set_cylinder_wall(0, 5.0, [0,0,0], [0,0,1], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))

        # 2. test a particle outside the wall
        self.system.particles[0].position = (10,0,0)
        # a. inside=False
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside=True
        self.ext_wall.set_cylinder_wall(0, 5.0, [0,0,0], [0,0,1], inside=True)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)


    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class cylinder_wall_convex_polyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polyhedron(seed=10)
        self.mc.shape_param.set('A', vertices = [(-0.5,-0.5,-0.5),
                                                (-0.5,0.5,-0.5),
                                                (-0.5,-0.5,0.5),
                                                (-0.5,0.5,0.5),
                                                (0.5,-0.5,-0.5),
                                                (0.5,0.5,-0.5),
                                                (0.5,-0.5,0.5),
                                                (0.5,0.5,0.5)])

        self.ext_wall = hpmc.field.wall(self.mc)
        self.ext_wall.add_cylinder_wall(5.0, [0,0,0], [0,0,1], inside=True)


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall, far from the boundary
        self.system.particles[0].position = (0,0,0)
        self.system.particles[0].orientation = (1,0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)
        # b. inside=False: then there should be an overlap
        self.ext_wall.set_cylinder_wall(0, 5.0, [0,0,0], [0,0,1], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))

        # 2. test a particle just overlapping the wall
        # a. inside=False: this is a pathological case wherein all vertices lie outside the cylinder, but
        # the cube still intersects the cylinder. count_overlaps should give 1 in this case (ie badly confined).
        # the intersection is between the face of the cube parallel to the yz axis, and the cylinder.
        for x in np.linspace(5.499999, 4.5, 1000):
            self.system.particles[0].position = (x,0,0)
            hoomd.util.quiet_status()
            self.assertEqual(self.ext_wall.count_overlaps(), 1)
            hoomd.util.unquiet_status()
        # b. inside=True: this should return an overlap.
        self.ext_wall.set_cylinder_wall(0, 5.0, [0,0,0], [0,0,1], inside=True)
        for x in np.linspace(4.475,5.475,1000):
            self.system.particles[0].position = (x,0,0)
            hoomd.util.quiet_status()
            self.assertEqual(self.ext_wall.count_overlaps(), 1)
            hoomd.util.unquiet_status()

        # 3. test a particle not quite overlapping the wall
        self.system.particles[0].position = (4.14,0,0)
        # a. inside=True
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside=False
        self.ext_wall.set_cylinder_wall(0, 5.0, [0,0,0], [0,0,1], inside=False)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

        # 4. test a particle outside the wall, far from the boundary
        self.system.particles[0].position = (10,0,0)
        # a. inside = False: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside = True: then there should be an overlap
        self.ext_wall.set_cylinder_wall(0, 5.0, [0,0,0], [0,0,1], inside=True)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)


    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class plane_wall_sphere_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.sphere(seed=10)
        self.mc.shape_param.set('A', diameter=1.0)

        self.ext_wall = hpmc.field.wall(self.mc)
        # this establishes the planar wall parallel to the yz plane, centered at (5,0,0).
        # the normal vector pointing along the (-) x axis means that
        # a particle to the left of the wall will be "inside" it, and
        # a particle to the right of the wall will be "outside" it.
        self.ext_wall.add_plane_wall([-1,0,0], [5,0,0])


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall
        self.system.particles[0].position = (0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)
        # b. inside=False: then there should be an overlap
        self.ext_wall.set_plane_wall(0, [1,0,0], [5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))

        # 2. test a particle outside the wall
        self.system.particles[0].position = (10,0,0)
        # a. inside=False
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside=True
        self.ext_wall.set_plane_wall(0, [-1,0,0],[5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)


    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class plane_wall_convex_polyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polyhedron(seed=10)
        self.mc.shape_param.set('A', vertices = [(-0.5,-0.5,-0.5),
                                                (-0.5,0.5,-0.5),
                                                (-0.5,-0.5,0.5),
                                                (-0.5,0.5,0.5),
                                                (0.5,-0.5,-0.5),
                                                (0.5,0.5,-0.5),
                                                (0.5,-0.5,0.5),
                                                (0.5,0.5,0.5)])

        self.ext_wall = hpmc.field.wall(self.mc)
        # this establishes the planar wall parallel to the yz plane, centered at (5,0,0).
        # the normal vector pointing along the (-) x axis means that
        # a particle to the left of the wall will be "inside" it, and
        # a particle to the right of the wall will be "outside" it.
        self.ext_wall.add_plane_wall([-1,0,0],[5,0,0])


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall, far from the boundary
        self.system.particles[0].position = (0,0,0)
        self.system.particles[0].orientation = (1,0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)
        # b. inside=False: then there should be an overlap
        self.ext_wall.set_plane_wall(0, [1,0,0], [5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))

        # 2. test a particle just overlapping the wall
        self.system.particles[0].position = (5.49,0,0)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        # b. inside=True: this should return an overlap.
        self.system.particles[0].position = (4.51,0,0)
        self.ext_wall.set_plane_wall(0, [-1,0,0],[5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

        # 3. test a particle not quite overlapping the wall
        self.system.particles[0].position = (4.14,0,0)
        # a. inside=True
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside=False
        self.ext_wall.set_plane_wall(0, [1,0,0], [5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

        # 4. test a particle outside the wall, far from the boundary
        self.system.particles[0].position = (10,0,0)
        # a. inside = False: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        # b. inside = True: then there should be an overlap
        self.ext_wall.set_plane_wall(0, [-1,0,0],[5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)


    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class plane_wall_convex_spheropolyhedron_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=10)
        self.mc.shape_param.set('A', vertices = [(-0.5,-0.5,-0.5),
                                                (-0.5,0.5,-0.5),
                                                (-0.5,-0.5,0.5),
                                                (-0.5,0.5,0.5),
                                                (0.5,-0.5,-0.5),
                                                (0.5,0.5,-0.5),
                                                (0.5,-0.5,0.5),
                                                (0.5,0.5,0.5)],
                                                sweep_radius=0.5)

        self.ext_wall = hpmc.field.wall(self.mc)
        # this establishes the planar wall parallel to the yz plane, centered at (5,0,0).
        # the normal vector pointing along the (-) x axis means that
        # a particle to the left of the wall will be "inside" it, and
        # a particle to the right of the wall will be "outside" it.
        self.ext_wall.add_plane_wall([-1,0,0],[5,0,0])


    def test(self):
        run(1, quiet=True)
        # 1. first test a particle within the wall, far from the boundary
        self.system.particles[0].position = (0,0,0)
        self.system.particles[0].orientation = (1,0,0,0)
        # a. inside = True: then there should be no overlaps
        self.assertEqual(self.ext_wall.count_overlaps(), 0)
        run(100)
        self.assertTrue(self.system.particles[0].position != (0,0,0))
        self.system.particles[0].position = (0,0,0)
        # b. inside=False: then there should be an overlap
        self.ext_wall.set_plane_wall(0, [1,0,0], [5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(self.system.particles[0].position == (0,0,0))

        # Test a particle that would overlap as just polyhedra
        self.system.particles[0].position = (5.49,0,0)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(
                np.allclose(
                    self.system.particles[0].position,
                    (5.49,0,0)
                    )
                )
        # Now test a particle that only overlaps with sweep radius
        self.system.particles[0].position = (5.99,0,0)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(
                np.allclose(
                    self.system.particles[0].position,
                    (5.99,0,0)
                    )
                )
        # b. inside=True: this should return an overlap.
        self.system.particles[0].position = (4.51,0,0)
        self.ext_wall.set_plane_wall(0, [-1,0,0],[5,0,0])
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(
                np.allclose(
                    self.system.particles[0].position,
                    (4.51,0,0)
                    )
                )

        self.system.particles[0].position = (4.01,0,0)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)
        run(100)
        self.assertTrue(
                np.allclose(
                    self.system.particles[0].position,
                    (4.01,0,0)
                    )
                )

    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

class sphere_wall_tetrahedron_specific_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=30, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polyhedron(seed=10)
        self.mc.shape_param.set('A', vertices = [( np.sqrt(3)/3.,  np.sqrt(3)/3.,  np.sqrt(3)/3.),
                                                (-np.sqrt(3)/3., -np.sqrt(3)/3.,  np.sqrt(3)/3.),
                                                ( np.sqrt(3)/3., -np.sqrt(3)/3., -np.sqrt(3)/3.),
                                                (-np.sqrt(3)/3.,  np.sqrt(3)/3., -np.sqrt(3)/3.)])

        self.ext_wall = hpmc.field.wall(self.mc)
        self.ext_wall.add_sphere_wall(0.2, origin=[0,0,0], inside=False)


    def test(self):
        run(1, quiet=True)
        # these are conditions that appear to result in an overlap in a test pos file.
        # however, the log file does not register any overlaps.
        self.system.particles[0].position = (-0.4134996, -0.30366718, 0.389844021)
        self.system.particles[0].orientation = (0.0164921231, 0.951677329, 0.197796895, -0.23433877)
        self.assertEqual(self.ext_wall.count_overlaps(), 1)

    def tearDown(self):
        del self.mc
        del self.system
        del self.ext_wall
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
