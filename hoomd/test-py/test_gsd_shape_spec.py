from hoomd import *
import hoomd
import unittest
import os
import tempfile
import numpy as np
from hoomd import _hoomd, hpmc, dem, md
import json

HPMC_CLASSES = ['sphere', 'ellipsoid', 'convex_polyhedron', \
                'convex_spheropolyhedron','polyhedron', \
                'convex_polygon', 'convex_spheropolygon', \
                'simple_polygon'];

DEM_CLASSES = ['WCA', 'SWCA'];

MD_CLASSES = ['gb'];

def parse_shape_spec(type_shapes):
    ret = [ json.loads(json_string) for json_string in type_shapes ];
    return ret;

class gsd_shape_spec_base(unittest.TestCase):

    def setUp(self):
        hoomd.context.initialize()
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.gsd');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    def setup_system(self, cls, shape_params, expected_shapespec, filename, dim):
        if dim == 2:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=5.50),n=5);
        elif dim == 3:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=5.50),n=5);
        snapshot = system.take_snapshot(all=True)
        bindex = np.random.choice(range(5**dim),int(0.5*5**dim),replace=False)
        if comm.get_rank() == 0:
        snapshot.particles.types = ['A', 'B']
        snapshot.particles.typeid[bindex] = 1

        hoomd.context.initialize()
        system = hoomd.init.read_snapshot(snapshot)
        if cls.__name__ in HPMC_CLASSES:
            obj = cls(seed=123);
            obj.shape_param.set('A', **shape_params["A"]);
            obj.shape_param.set('B', **shape_params["B"]);
        else:
            nl = md.nlist.cell();
            md.integrate.mode_standard(dt=0.001);
            if cls.__name__ in DEM_CLASSES:
                obj = cls(nlist=nl, radius=.5);
                obj.setParams('A', **shape_params["A"]);
                obj.setParams('B', **shape_params["B"]);
            elif cls.__name__ in MD_CLASSES:
                obj = cls(r_cut=3, nlist=nl)
                obj.pair_coeff.set('A', 'A', **shape_params['A'])
                obj.pair_coeff.set('B', 'B', **shape_params['B'])
                obj.pair_coeff.set('A', 'B', **shape_params['B'])
            md.integrate.nvt(group=group.all(), kT=1.0, tau=0.5)
        dumper = dump.gsd(filename=filename, group=group.all(), period=1, overwrite=True);
        dumper.dump_shape(obj);
        steps = 5
        hoomd.run(steps);
        if comm.get_rank() == 0:
            reader = _hoomd.GSDReader(hoomd.context.exec_conf, filename, 0, False);
            for i in range(steps):
                shape_spec = parse_shape_spec(reader.readTypeShapesPy(i));
                self.assertEqual(shape_spec[0], expected_shapespec[0]);
                self.assertEqual(shape_spec[1], expected_shapespec[1]);

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);
        comm.barrier_all();

class hpmc_gsd_shape_spec(gsd_shape_spec_base):

    def test_sphere(self):
        expected_shapespec = [dict(type='Sphere',diameter=1), dict(type='Sphere',diameter=2)];
        shape_params = dict(A=dict(diameter=1), B=dict(diameter=2));
        self.setup_system(cls=hpmc.integrate.sphere, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_ellipsoid(self):
        shape_params = dict(A=dict(a=1.2, b=1.5, c=2), B=dict(a=2.3, b=1.5, c=1.7));
        expected_shapespec = [dict(type='Ellipsoid', **shape_params["A"]), \
                              dict(type='Ellipsoid', **shape_params["B"])];
        self.setup_system(cls=hpmc.integrate.ellipsoid, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_convex_polyhedron(self):
        cube_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], \
                      [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]];
        tetra_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        shape_params = dict(A=dict(vertices=cube_verts), B=dict(vertices=tetra_verts));
        expected_shapespec = [dict(type='ConvexPolyhedron', rounding_radius=0, **shape_params["A"]), \
                              dict(type='ConvexPolyhedron', rounding_radius=0, **shape_params["B"])];
        print(expected_shapespec)
        self.setup_system(cls=hpmc.integrate.convex_polyhedron, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_convex_spheropolyhedron(self):
        cube_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], \
                      [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]];
        tetra_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        shape_params = dict(A=dict(vertices=cube_verts,  sweep_radius=0.1), \
                            B=dict(vertices=tetra_verts, sweep_radius=0.2));
        expected_shapespec = [dict(type='ConvexPolyhedron', rounding_radius=0.1, vertices=cube_verts), \
                              dict(type='ConvexPolyhedron', rounding_radius=0.2, vertices=tetra_verts)];
        self.setup_system(cls=hpmc.integrate.convex_spheropolyhedron, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_general_polyhedron(self):
        cube_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], \
                      [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]];
        cube_faces = [[0,2,6], [6,4,0], [5,0,4], [5,1,0], [5,4,6], [5,6,7], [3,2,0], [3,0,1], \
                      [3,6,2], [3,7,6], [3,1,5], [3,5,7]]
        tetra_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        tetra_faces = [[0,1,2], [3,0,2], [3,2,1], [3,1,0]];
        shape_params = dict(A=dict(vertices=cube_verts, faces=cube_faces), \
                            B=dict(vertices=tetra_verts, faces=tetra_faces));
        expected_shapespec = [dict(type='Mesh', vertices=cube_verts, indices=cube_faces), \
                              dict(type='Mesh', vertices=tetra_verts, indices=tetra_faces)];
        self.setup_system(cls=hpmc.integrate.polyhedron, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_convex_polygon(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        trg_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=trg_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0, vertices=sq_verts), \
                              dict(type='Polygon', rounding_radius=0, vertices=trg_verts)];
        self.setup_system(cls=hpmc.integrate.convex_polygon, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_convex_spheropolygon(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        trg_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]];
        shape_params = dict(A=dict(vertices=sq_verts, sweep_radius=0.3), \
                            B=dict(vertices=trg_verts, sweep_radius=0.13));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.3, vertices=sq_verts), \
                              dict(type='Polygon', rounding_radius=0.13, vertices=trg_verts)];
        self.setup_system(cls=hpmc.integrate.convex_spheropolygon, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_convex_spheropolygon_with_sphere(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        sph_verts = [[0.0, 0.0]];
        shape_params = dict(A=dict(vertices=sq_verts, sweep_radius=0.3), \
                            B=dict(vertices=sph_verts, sweep_radius=1.2));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.3, vertices=sq_verts), \
                              dict(type='Sphere', diameter=2*shape_params["B"]["sweep_radius"])];
        self.setup_system(cls=hpmc.integrate.convex_spheropolygon, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_simple_polygon(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        concave_verts = [[-0.5, 0.0], [0.5, -0.5], [0.0, 0.0], [0.5, 0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=concave_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0, vertices=sq_verts), \
                              dict(type='Polygon', rounding_radius=0, vertices=concave_verts)];
        self.setup_system(cls=hpmc.integrate.simple_polygon, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

class dem_gsd_shape_spec(gsd_shape_spec_base):

    def test_wca_2d(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        trg_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=trg_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.5, vertices=sq_verts), \
                              dict(type='Polygon', rounding_radius=0.5, vertices=trg_verts)];
        self.setup_system(cls=dem.pair.WCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_wca_2d_with_sphere(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        sph_verts = [[-0.5, -0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=sph_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.5, vertices=sq_verts), \
                              dict(type='Sphere', diameter=1)];
        self.setup_system(cls=dem.pair.WCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_wca_3d(self):
        cube_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], \
                      [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]];
        cube_faces = [[0,2,6], [6,4,0], [5,0,4], [5,1,0], [5,4,6], [5,6,7], [3,2,0], [3,0,1], \
                      [3,6,2], [3,7,6], [3,1,5], [3,5,7]]
        tetra_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        tetra_faces = [[0,1,2], [3,0,2], [3,2,1], [3,1,0]];
        shape_params = dict(A=dict(vertices=cube_verts, faces=cube_faces), \
                            B=dict(vertices=tetra_verts, faces=tetra_faces));
        expected_shapespec = [dict(type='ConvexPolyhedron', rounding_radius=0.5, vertices=cube_verts), \
                              dict(type='ConvexPolyhedron', rounding_radius=0.5, vertices=tetra_verts)];
        self.setup_system(cls=dem.pair.WCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_wca_3d_with_sphere(self):
        cube_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], \
                      [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]];
        cube_faces = [[0,2,6], [6,4,0], [5,0,4], [5,1,0], [5,4,6], [5,6,7], [3,2,0], [3,0,1], \
                      [3,6,2], [3,7,6], [3,1,5], [3,5,7]]
        sph_verts = [[-0.5, -0.5, 0.5]];
        shape_params = dict(A=dict(vertices=cube_verts, faces=cube_faces), \
                            B=dict(vertices=sph_verts, faces=[]));
        expected_shapespec = [dict(type='ConvexPolyhedron', rounding_radius=0.5, vertices=cube_verts), \
                              dict(type='Sphere', diameter=1)];
        self.setup_system(cls=dem.pair.WCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_swca_2d(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        trg_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=trg_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.5, vertices=sq_verts), \
                              dict(type='Polygon', rounding_radius=0.5, vertices=trg_verts)];
        self.setup_system(cls=dem.pair.SWCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_swca_2d_with_sphere(self):
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        sph_verts = [[-0.5, -0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=sph_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.5, vertices=sq_verts), \
                              dict(type='Sphere', diameter=1)];
        self.setup_system(cls=dem.pair.SWCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_swca_3d(self):
        cube_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], \
                      [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]];
        cube_faces = [[0,2,6], [6,4,0], [5,0,4], [5,1,0], [5,4,6], [5,6,7], [3,2,0], [3,0,1], \
                      [3,6,2], [3,7,6], [3,1,5], [3,5,7]]
        tetra_verts = [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]];
        tetra_faces = [[0,1,2], [3,0,2], [3,2,1], [3,1,0]];
        shape_params = dict(A=dict(vertices=cube_verts, faces=cube_faces), \
                            B=dict(vertices=tetra_verts, faces=tetra_faces));
        expected_shapespec = [dict(type='ConvexPolyhedron', rounding_radius=0.5, vertices=cube_verts), \
                              dict(type='ConvexPolyhedron', rounding_radius=0.5, vertices=tetra_verts)];
        self.setup_system(cls=dem.pair.SWCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

    def test_swca_3d_with_sphere(self):
        cube_verts = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5], [-0.5, 0.5, 0.5], \
                      [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]];
        cube_faces = [[0,2,6], [6,4,0], [5,0,4], [5,1,0], [5,4,6], [5,6,7], [3,2,0], [3,0,1], \
                      [3,6,2], [3,7,6], [3,1,5], [3,5,7]]
        sph_verts = [[-0.5, -0.5, 0.5]];
        shape_params = dict(A=dict(vertices=cube_verts, faces=cube_faces), \
                            B=dict(vertices=sph_verts, faces=[]));
        expected_shapespec = [dict(type='ConvexPolyhedron', rounding_radius=0.5, vertices=cube_verts), \
                              dict(type='Sphere', diameter=1)];
        self.setup_system(cls=dem.pair.SWCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);

class md_gsd_shape_spec(gsd_shape_spec_base):

    def test_gay_berne(self):
        shape_params = dict(A=dict(epsilon=1, lperp=0.5, lpar=1), \
                            B=dict(epsilon=2, lperp=0.5, lpar=1));
        expected_shapespec = [dict(type='Ellipsoid', a=shape_params['A']['lperp'], \
                                   b=shape_params['A']['lperp'], \
                                   c=shape_params['A']['lpar']), \
                              dict(type='Ellipsoid', a=shape_params['B']['lperp'], \
                                   b=shape_params['B']['lperp'], \
                                   c=shape_params['B']['lpar']) ];
        self.setup_system(cls=md.pair.gb, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
