from hoomd import *
import hoomd
import unittest
import os
import tempfile
import numpy as np
from hoomd import _hoomd, dem, md
import json

def parse_shape_spec(type_shapes):
    ret = [ json.loads(json_string) for json_string in type_shapes ];
    return ret;

class dem_shape_spec_base(unittest.TestCase):

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
        system = hoomd.init.read_snapshot(snapshot);
        md.integrate.mode_standard(dt=0.001);
        obj = cls(nlist=md.nlist.cell(), radius=.5);
        obj.setParams('A', **shape_params["A"]);
        obj.setParams('B', **shape_params["B"]);
        md.integrate.nvt(group=group.all(), kT=1.0, tau=0.5)
        dumper = dump.gsd(filename=filename, group=group.all(), period=1, overwrite=True);
        dumper.dump_shape(obj);
        steps = 5
        hoomd.run(steps);
        reader = _hoomd.GSDReader(hoomd.context.exec_conf, filename, 0, False);
        if comm.get_rank() == 0:
            for i in range(steps):
                shape_spec = parse_shape_spec(reader.readTypeShapesPy(i));
                self.assertEqual(shape_spec[0], expected_shapespec[0]);
                self.assertEqual(shape_spec[1], expected_shapespec[1]);

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
        if comm.get_num_ranks() > 1:
            return
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        trg_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=trg_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.5, vertices=sq_verts), \
                              dict(type='Polygon', rounding_radius=0.5, vertices=trg_verts)];
        self.setup_system(cls=dem.pair.SWCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_swca_2d_with_sphere(self):
        if comm.get_num_ranks() > 1:
            return
        sq_verts = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        sph_verts = [[-0.5, -0.5]];
        shape_params = dict(A=dict(vertices=sq_verts), \
                            B=dict(vertices=sph_verts));
        expected_shapespec = [dict(type='Polygon', rounding_radius=0.5, vertices=sq_verts), \
                              dict(type='Sphere', diameter=1)];
        self.setup_system(cls=dem.pair.SWCA, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=2);

    def test_swca_3d(self):
        if comm.get_num_ranks() > 1:
            return
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
        if comm.get_num_ranks() > 1:
            return
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

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);
        comm.barrier_all();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
