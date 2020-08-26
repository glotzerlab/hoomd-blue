import math
import sys
import json
import copy
import numpy as np
import copy
import time
import unittest
from plato import geometry
ConvexHull = None;
library = None;
lattice = None;
trianglehedra = None;
try:
    from pyhull.convex_hull import ConvexHull
except ImportError:
    ConvexHull = None;
    library = None;
    lattice = None;
    trianglehedra = None;

import hoomd
from hoomd import hpmc

# In the following face is assumed to be an array of indices of a triangular face.
# verts is a numpy array of the vertices
def getOutwardNormal(verts, face, thresh = 0.0001):
    assert(len(face) == 3);
    (a, b, c) = verts[face];
    n = np.cross((b - a),(c - a));
    flip = False;
    for k in range(len(verts)):
        if (not k in face):
            d = n.dot(verts[k] - a);
            if abs(d) > thresh and d > 0: # by convexity
                flip = True;
                break;
    if flip:
        n = -n;
    return n;

def sortFaces(verts, faces):
    sorted_face = [];
    for face in faces:
        nout = getOutwardNormal(verts, face);
        (a, b, c) = verts[face];
        n = np.cross((b - a),(c - a));
        if n.dot(nout) > 0:
            sorted_face.append(face);
        else:
            sorted_face.append(face[::-1]);
    return np.array(sorted_face)

class mass_properties_convex_polyhedron_test(unittest.TestCase):
    def setUp(self):
        np.random.seed(1001248987);
        hoomd.context.initialize();
        hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=1.0),n=[1,1,1]);


    def test_convex_polyhedron(self):
        print("")
        cpp_time = 0.0;
        py_time = 0.0;
        for _ in range(100):
            nverts = np.random.randint(10, 128);
            make_verts = hpmc._hpmc.make_poly3d_verts;
            mass_class = hpmc._hpmc.MassPropertiesConvexPolyhedron;
            verts = 5.0*np.random.rand(nverts, 3);

            start = time.time();
            hull = ConvexHull(verts);
            faces = np.array(hull.vertices);
            faces = sortFaces(verts, faces);
            vol, com, inertia = geometry.massProperties(verts, faces);
            end = time.time();
            py_time += end-start;
            py_verts = [] # the actual points in the convex_hull
            for i, j, k in hull.vertices:
                py_verts.append(list(verts[i]))
                py_verts.append(list(verts[j]))
                py_verts.append(list(verts[k]))
            py_verts = np.asarray(sorted(py_verts,
                                         key=lambda x: x[0]**2 + x[1]**2 + x[2]**2))
            start = time.time();
            mp = mass_class(make_verts(verts.tolist(), 0, False, hoomd.context.current.system_definition.getParticleData().getExecConf()));
            end = time.time();
            cpp_time += end-start;

            cpp_volume = mp.volume()
            cpp_com = [ mp.center_of_mass(i) for i in range(3) ];
            cpp_inertia = [ mp.moment_of_inertia(i) for i in range(6) ];

            cpp_verts = [] # the actual points in the convex_hull
            for f in range(mp.num_faces()):
                for i in range(3):
                    cpp_verts.append(mp.vertices(f, i))
                # cpp_ids.update([mp.index(f, i) for i in range(3)]);

            # faces my be differnt because triangulation is not unique but there should be
            # the same number of faces and the same vertices will be in the hull.
            # Also test the result gives us the same result for the volume, inertia and com
            self.assertEqual(len(faces), mp.num_faces());
            # test the vertices are the same.
            cpp_verts = np.asarray(sorted(cpp_verts,
                                          key=lambda x: x[0]**2 + x[1]**2 + x[2]**2))
            cpp_verts = np.unique(cpp_verts, axis=0)
            py_verts = np.unique(py_verts, axis=0)
            np.testing.assert_allclose(cpp_verts, py_verts)
            # volume is equal
            self.assertAlmostEqual(vol, cpp_volume, 5);
            # com is equal
            self.assertAlmostEqual(com[0], cpp_com[0], 5);
            self.assertAlmostEqual(com[1], cpp_com[1], 5);
            self.assertAlmostEqual(com[2], cpp_com[2], 5);
            # interia tensor is the same.
            self.assertAlmostEqual(inertia[0], cpp_inertia[0], 4); # xx
            self.assertAlmostEqual(inertia[3], cpp_inertia[1], 4); # yy
            self.assertAlmostEqual(inertia[5], cpp_inertia[2], 4); # zz
            self.assertAlmostEqual(inertia[1], cpp_inertia[3], 4); # xy
            self.assertAlmostEqual(inertia[4], cpp_inertia[4], 4); # yz
            self.assertAlmostEqual(inertia[2], cpp_inertia[5], 4); # xz
        print("c++ ran 100 convex hulls in {} ({} per call)".format(cpp_time, cpp_time/100));
        print("py ran 100 convex hulls in {} ({} per call)".format(py_time, py_time/100));

    def tearDown(self):
        pass;


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
