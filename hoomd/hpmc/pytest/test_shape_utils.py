import numpy as np
import time
import pytest
import coxeter
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


def getOutwardNormal(verts, face, thresh=0.0001):
    assert(len(face) == 3)
    (a, b, c) = verts[face]
    n = np.cross((b - a), (c - a))
    flip = False
    for k in range(len(verts)):
        if (not k in face):
            d = n.dot(verts[k] - a)
            if abs(d) > thresh and d > 0:  # by convexity
                flip = True
                break
    if flip:
        n = -n
    return n


def sortFaces(verts, faces):
    sorted_face = []
    for face in faces:
        nout = getOutwardNormal(verts, face)
        (a, b, c) = verts[face]
        n = np.cross((b - a), (c - a))
        if n.dot(nout) > 0:
            sorted_face.append(face)
        else:
            sorted_face.append(face[::-1])
    return np.array(sorted_face)


def test_mass_properties():
    cpp_time = 0.0
    py_time = 0.0
    for _ in range(100):
        # nverts = np.random.randint(10, 128)
        nverts =16
        make_verts = hpmc._hpmc.PolyhedronVertices
        mass_class = hpmc._hpmc.MassPropertiesConvexPolyhedron
        verts = 5.0 * np.random.rand(nverts, 3)
        hull = ConvexHull(verts);
        faces = np.array(hull.vertices);
        faces = sortFaces(verts, faces);
        vol, com, inertia = geometry.massProperties(verts, faces);

        start = time.time()
        py_shape = coxeter.shapes.ConvexPolyhedron(verts)
        py_shape.diagonalize_inertia()
        end = time.time()
        py_time += end - start
        py_verts = [] # the actual points in the convex_hull
        for i, j, k in hull.vertices:
            py_verts.append(list(verts[i]))
            py_verts.append(list(verts[j]))
            py_verts.append(list(verts[k]))
        py_verts = np.asarray(sorted(py_verts,
                                     key=lambda x: x[0]**2 + x[1]**2 + x[2]**2))
        # py_verts = np.asarray(sorted(list(py_shape.vertices),
        #                              key=lambda x: x[0]**2 + x[1]**2 + x[2]**2))
        start = time.time()
        mp = mass_class(make_verts({'vertices': verts, 'sweep_radius': 0.0, 'ignore_statistics': 0}))
        end = time.time()
        cpp_time += end - start

        cpp_volume = mp.volume()
        cpp_com = [mp.center_of_mass(i) for i in range(3)]
        cpp_inertia = [mp.moment_of_inertia(i) for i in range(6)]

        cpp_verts = []  # the actual points in the convex_hull
        for f in range(mp.num_faces()):
            for i in range(3):
                cpp_verts.append(mp.vertices(f, i))
            # cpp_ids.update([mp.index(f, i) for i in range(3)]);

        # faces may be different because triangulation is not unique but there should be
        # the same number of faces and the same vertices will be in the hull.
        # Also test the result gives us the same result for the volume, inertia and com
        np.testing.assert_allclose(py_shape.num_faces, mp.num_faces())
        # test the vertices are the same.
        cpp_verts = np.asarray(sorted(cpp_verts,
                                      key=lambda x: x[0]**2 + x[1]**2 + x[2]**2))
        cpp_verts = np.unique(cpp_verts, axis=0)
        py_verts = np.unique(py_verts, axis=0)
        print(py_shape.num_vertices)
        np.testing.assert_allclose(cpp_verts, py_verts)
        np.testing.assert_allclose(cpp_volume, py_shape.volume, rtol=0.2)
        np.testing.assert_allclose(cpp_com, py_shape.center, rtol=0.2)
        np.testing.assert_allclose(cpp_inertia, py_shape.inertia_tensor, rtol=0.2)
        np.testing.assert_allclose(cpp_volume, vol)
        np.testing.assert_allclose(cpp_com, com)
        tmp = inertia
        inertia[1] = tmp[3]
        inertia[2] = tmp[5]
        inertia[3] = tmp[1]
        inertia[5] = tmp[2]
        # np.testing.assert_allclose(cpp_inertia, inertia)
    print("c++ ran 100 convex hulls in {} ({} per call)".format(cpp_time, cpp_time / 100))
    print("py ran 100 convex hulls in {} ({} per call)".format(py_time, py_time / 100))
