import hoomd
from hoomd import hpmc
import unittest
hoomd.context.initialize()
print(hoomd.__file__)

class test_type_shapes(unittest.TestCase):
    def setUp(self):
        hoomd.context.initialize()

    def test_type_shapes_convex_polygon(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.convex_polygon(seed=10);
        test_verts = [(1, 0), (0, 1), (-1, -1)]
        self.mc.shape_param.set('A', vertices=test_verts)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Polygon')
        self.assertEqual(len(shape_types[0]['vertices']), 3)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))
        self.assertTrue(shape_types[0]['rounding_radius'] == 0 )

    def test_type_shapes_simple_polygon(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.simple_polygon(seed=10);
        test_verts = [(1, 0), (0, 1), (-1, -1)]
        self.mc.shape_param.set('A', vertices=test_verts)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Polygon')
        self.assertEqual(len(shape_types[0]['vertices']), 3)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))
        self.assertTrue(shape_types[0]['rounding_radius'] == 0 )

    def test_type_shapes_disks(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.sphere(seed=10);
        test_diam = 1
        self.mc.shape_param.set('A', diameter = test_diam)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Sphere')
        self.assertEqual(shape_types[0]['diameter'], test_diam)
        self.assertNotIn('vertices', shape_types[0])

    def test_type_shapes_spheres(self):
        box = hoomd.data.boxdim(10, dimensions=3)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.sphere(seed=10);
        test_diam = 1
        self.mc.shape_param.set('A', diameter = test_diam)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Sphere')
        self.assertEqual(shape_types[0]['diameter'], test_diam)
        self.assertNotIn('vertices', shape_types[0])

    def test_type_shapes_convex_polyhedron(self):
        box = hoomd.data.boxdim(10, dimensions=3)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.convex_polyhedron(seed=10);
        test_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], \
                      [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        self.mc.shape_param.set('A', vertices=test_verts)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'ConvexPolyhedron')
        self.assertEqual(len(shape_types[0]['vertices']), 4)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))
        self.assertTrue(shape_types[0]['rounding_radius'] == 0 )

    def test_type_shapes_ellipsoid(self):
        box = hoomd.data.boxdim(10, dimensions=3)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.ellipsoid(seed=10);
        test_abc = {"a":1.0, "b":2.0, "c":3.0}
        self.mc.shape_param.set('A', **test_abc)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Ellipsoid')
        for k in test_abc.keys():
            self.assertAlmostEqual(shape_types[0][k], test_abc[k])
        self.assertNotIn('vertices', shape_types[0])

    def test_type_shapes_convex_spheropolyhedron(self):
        box = hoomd.data.boxdim(10, dimensions=3)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=10);
        test_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], \
                      [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        self.mc.shape_param.set('A', vertices=test_verts, sweep_radius=0.3)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'ConvexPolyhedron')
        self.assertEqual(len(shape_types[0]['vertices']), 4)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))
        self.assertTrue(shape_types[0]['rounding_radius'] == 0.3 )

    def test_type_shapes_convex_spheropolygon(self):
        box = hoomd.data.boxdim(10, dimensions=2)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.convex_spheropolygon(seed=10);
        test_verts = [(1, 0), (0, 1), (-1, -1)]
        self.mc.shape_param.set('A', vertices=test_verts, sweep_radius=0.3)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Polygon')
        self.assertEqual(len(shape_types[0]['vertices']), 3)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))
        self.assertTrue(shape_types[0]['rounding_radius'] == 0.3 )

    def test_type_shapes_polyhedron(self):
        box = hoomd.data.boxdim(10, dimensions=3)
        snap = hoomd.data.make_snapshot(N=2, box=box)
        snap.particles.types = ['A']
        self.system = hoomd.init.read_snapshot(snap)

        self.mc = hpmc.integrate.polyhedron(seed=10);
        test_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], \
                      [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        test_faces = [[0,1,2], [3,0,2], [3,2,1], [3,1,0]];
        self.mc.shape_param.set('A', vertices=test_verts, faces=test_faces)

        shape_types = self.mc.get_type_shapes()

        self.assertEqual(shape_types[0]['type'], 'Mesh')
        self.assertEqual(len(shape_types[0]['vertices']), 4)
        self.assertEqual(len(shape_types[0]['indices']), 4)
        self.assertTrue(all([shape_types[0]['vertices'][i] == list(test_verts[i]) for i in range(len(test_verts))]))
        self.assertTrue(all([shape_types[0]['indices'][i] == list(test_faces[i]) for i in range(len(test_faces))]))

    def tearDown(self):
        del self.mc
        del self.system
        hoomd.context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
