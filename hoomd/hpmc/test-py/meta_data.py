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

# Very basic tests to ensure that meta data output does not create any errors
# and is producing expected results.

class convex_polygon_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polygon(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        vertices = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        self.mc.shape_param.set('A',  vertices=vertices)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.convex_polygon', meta_data)
        self.assertNotEqual(meta_data['hoomd.hpmc.integrate.convex_polygon'], None)
        self.assertEqual(meta_data['hoomd.hpmc.integrate.convex_polygon']['shape_param']['A']['vertices'], vertices)

class simple_polygon_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.simple_polygon(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        vertices = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        self.mc.shape_param.set('A',  vertices=vertices)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.simple_polygon', meta_data)
        self.assertEqual(meta_data['hoomd.hpmc.integrate.simple_polygon']['shape_param']['A']['vertices'], vertices)

class convex_polyhedron_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.convex_polyhedron(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        vertices=[[-2,-1,-1],
                   [-2,1,-1],
                   [-2,-1,1],
                   [-2,1,1],
                   [2,-1,-1],
                   [2,1,-1],
                   [2,-1,1],
                   [2,1,1]]
        self.mc.shape_param.set('A',  vertices=vertices)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.convex_polyhedron', meta_data)
        self.assertEqual(meta_data['hoomd.hpmc.integrate.convex_polyhedron']['shape_param']['A']['vertices'], vertices)

class sphere_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.sphere(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

    def test_metadata_dump(self):
        self.mc.shape_param.set('A', diameter=1.0)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.sphere', meta_data)
        self.assertEqual(meta_data['hoomd.hpmc.integrate.sphere']['shape_param']['A']['diameter'], 1.0)

class sphere_union_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.sphere_union(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

    def test_metadata_dump(self):
        diameters = [1.0, 1.0]
        centers = [[-0.25, 0, 0], [0.25, 0, 0]]
        self.mc.shape_param.set('A', diameters=diameters, centers=centers)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.sphere_union', meta_data)
        # readback is not currently enabled for shape union
        # self.assertEqual(meta_data['hoomd.hpmc.integrate.sphere_union']['shape_param']['A']['diameters'], diameters)
        # self.assertEqual(meta_data['hoomd.hpmc.integrate.sphere_union']['shape_param']['A']['centers'], centers)

class faceted_ellipsoid_union_test(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=3), particle_types=['A'])
        self.mc = hpmc.integrate.faceted_ellipsoid_union(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

    def test_metadata_dump(self):
        self.mc.shape_param.set('A', normals=[[(1,0,0)],[(0,0,1)]], offsets=[[-0.2],[-0.3]],
            vertices=[[],[]], origins=[(0,0,0),(0,0,0)],
            axes=[(0.5,0.5,0.5),(0.5,0.5,1)],centers=[(-0.25, 0, 0), (0.25, 0, 0)],
            orientations=[(1,0,0,0),(0,0,0,1)], capacity=16);

        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.faceted_ellipsoid_union', meta_data)

class convex_spheropolygon_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.convex_spheropolygon(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        vertices=[[-2,-1],
                   [-2,1],
                   [-2,-1],
                   [-2,1],
                   [2,-1],
                   [2,1],
                   [2,-1],
                   [2,1]]

        self.mc.shape_param.set('A',  vertices=vertices)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.convex_spheropolygon', meta_data)
        self.assertEqual(meta_data['hoomd.hpmc.integrate.convex_spheropolygon']['shape_param']['A']['vertices'], vertices)

class polyhedron_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.polyhedron(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        vertices=[[-2,-1,-1],
                   [-2,1,-1],
                   [-2,-1,1],
                   [-2,1,1],
                   [2,-1,-1],
                   [2,1,-1],
                   [2,-1,1],
                   [2,1,1]]
        faces = [[1,2,3,4]]
        self.mc.shape_param.set('A', vertices=vertices, faces=[])
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.polyhedron', meta_data)
        self.assertEqual(meta_data['hoomd.hpmc.integrate.polyhedron']['shape_param']['A']['vertices'], vertices)

# apparently not all params are being set here, but I do not see proper documentation
# so it is just commented out until it is updated by the author
class faceted_sphere_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.faceted_sphere(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

    def test_metadata_dump(self):
        shape_param = dict(
            normals=[[-1,0,0],
              [1,0,0],
              [0,1,0,],
              [0,-1,0],
              [0,0,1],
              [0,0,-1]],
            offsets=[-1]*6,
            vertices=[[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]],
            diameter=2,
            origin=[0,0,0]
            )
        self.mc.shape_param.set('A',  ** shape_param)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.faceted_sphere', meta_data)
        for key in shape_param:
            self.assertEqual(meta_data['hoomd.hpmc.integrate.faceted_sphere']['shape_param']['A'][key], shape_param[key])

class convex_spheropolyhedron_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        vertices=[[-2,-1,-1],
                   [-2,1,-1],
                   [-2,-1,1],
                   [-2,1,1],
                   [2,-1,-1],
                   [2,1,-1],
                   [2,-1,1],
                   [2,1,1]]
        self.mc.shape_param.set('A',  vertices=vertices)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.convex_spheropolyhedron', meta_data)
        self.assertEqual(meta_data['hoomd.hpmc.integrate.convex_spheropolyhedron']['shape_param']['A']['vertices'], vertices)

class ellipsoid_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.ellipsoid(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        shape_param = dict(a=0.5, b=0.25, c=0.125)
        self.mc.shape_param.set('A',  **shape_param)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.ellipsoid', meta_data)
        for key in shape_param:
            self.assertAlmostEqual(meta_data['hoomd.hpmc.integrate.ellipsoid']['shape_param']['A'][key], shape_param[key]) # using almost equal now because storing the data in C++ gives us finite precision.

class sphinx_test(unittest.TestCase):

    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=10, dimensions=2), particle_types=['A'])
        self.mc = hpmc.integrate.sphinx(seed=10);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

    def test_metadata_dump(self):
        shape_param = dict(diameters=[2,-2.2,-2.2], centers=[(0,0,0), (0,0,1.15), (0,0,-1.15)], \
                           colors=['ff','ffff00','ffff00'])
        self.mc.shape_param.set('A',  **shape_param)
        context.current.sorter.set_params(grid=8)
        meta_data = meta.dump_metadata()
        self.assertIn('hoomd.hpmc.integrate.sphinx', meta_data)

        for i,d in enumerate(shape_param['diameters']):
            self.assertAlmostEqual(meta_data['hoomd.hpmc.integrate.sphinx']['shape_param']['A']['diameters'][i], d );

        for i,center in enumerate(shape_param['centers']):
            for j,c in enumerate(center):
                self.assertAlmostEqual(meta_data['hoomd.hpmc.integrate.sphinx']['shape_param']['A']['centers'][i][j], c);
        self.assertEqual(meta_data['hoomd.hpmc.integrate.sphinx']['shape_param']['A']['colors'], shape_param['colors']);


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
