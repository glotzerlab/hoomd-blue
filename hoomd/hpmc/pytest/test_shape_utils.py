# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest
import coxeter
from plato import geometry
from pyhull.convex_hull import ConvexHull
import hoomd
from hoomd import hpmc
from copy import deepcopy


def get_outward_normal(verts, face, thresh=0.0001):
    assert (len(face) == 3)
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


def sort_faces(verts, faces):
    sorted_face = []
    for face in faces:
        nout = get_outward_normal(verts, face)
        (a, b, c) = verts[face]
        n = np.cross((b - a), (c - a))
        if n.dot(nout) > 0:
            sorted_face.append(face)
        else:
            sorted_face.append(face[::-1])
    return np.array(sorted_face)


def _vertices():
    platonic_shapes = [
        "Cube", "Tetrahedron", "Octahedron", "Icosahedron", "Dodecahedron"
    ]
    verts = []
    names = []
    for shape in platonic_shapes:
        verts.append(
            np.asarray(
                coxeter.families.PlatonicFamily.get_shape(shape).vertices))
        names.append(shape)
    for i in range(20):
        verts.append(np.random.rand(np.random.randint(10, 128), 3))
        names.append("random" + str(i))
    return zip(names, verts)


@pytest.fixture(scope="function", params=_vertices(), ids=(lambda x: x[0]))
def vertices(request):
    return deepcopy(request.param)


_make_verts = hpmc._hpmc.PolyhedronVertices
_mass_class = hpmc._hpmc.MassPropertiesConvexPolyhedron


def test_convex_hull_vertices(vertices):
    _, verts = vertices

    hull = ConvexHull(verts)
    py_verts = []
    for i, j, k in hull.vertices:
        py_verts.append(list(verts[i]))
        py_verts.append(list(verts[j]))
        py_verts.append(list(verts[k]))
    py_verts = np.asarray(
        sorted(py_verts, key=lambda x: x[0]**2 + x[1]**2 + x[2]**2))
    py_verts = np.unique(py_verts, axis=0)

    mp = _mass_class(
        _make_verts({
            'vertices': verts,
            'sweep_radius': 0.0,
            'ignore_statistics': 0
        }))
    cpp_verts = []
    for f in range(mp.num_faces()):
        for i in range(3):
            cpp_verts.append(mp.vertices(f, i))
    cpp_verts = np.asarray(
        sorted(cpp_verts, key=lambda x: x[0]**2 + x[1]**2 + x[2]**2))
    cpp_verts = np.unique(cpp_verts, axis=0)

    np.testing.assert_allclose(cpp_verts, py_verts)


def test_num_faces(vertices):
    _, verts = vertices

    hull = ConvexHull(verts)
    faces = np.array(hull.vertices)
    faces = sort_faces(verts, faces)

    mp = _mass_class(
        _make_verts({
            'vertices': verts,
            'sweep_radius': 0.0,
            'ignore_statistics': 0
        }))

    np.testing.assert_allclose(mp.num_faces(), len(faces))


def test_volume(vertices):
    _, verts = vertices

    hull = ConvexHull(verts)
    faces = np.array(hull.vertices)
    faces = sort_faces(verts, faces)
    py_vol, _, _ = geometry.massProperties(verts, faces)

    mp = _mass_class(
        _make_verts({
            'vertices': verts,
            'sweep_radius': 0.0,
            'ignore_statistics': 0
        }))

    np.testing.assert_allclose(mp.volume(), py_vol, rtol=1e-5, atol=1e-5)


def test_center_of_mass(vertices):
    _, verts = vertices

    hull = ConvexHull(verts)
    faces = np.array(hull.vertices)
    faces = sort_faces(verts, faces)
    _, py_com, _ = geometry.massProperties(verts, faces)

    mp = _mass_class(
        _make_verts({
            'vertices': verts,
            'sweep_radius': 0.0,
            'ignore_statistics': 0
        }))
    cpp_com = [mp.center_of_mass(i) for i in range(3)]

    np.testing.assert_allclose(cpp_com, py_com, rtol=1e-5, atol=1e-5)


def test_inertia(vertices):
    _, verts = vertices

    hull = ConvexHull(verts)
    faces = np.array(hull.vertices)
    faces = sort_faces(verts, faces)
    _, _, py_inertia = geometry.massProperties(verts, faces)
    tmp = py_inertia + 0.0
    py_inertia[1] = tmp[3]
    py_inertia[2] = tmp[5]
    py_inertia[3] = tmp[1]
    py_inertia[5] = tmp[2]

    mp = _mass_class(
        _make_verts({
            'vertices': verts,
            'sweep_radius': 0.0,
            'ignore_statistics': 0
        }))
    cpp_inertia = [mp.moment_of_inertia(i) for i in range(6)]

    np.testing.assert_allclose(cpp_inertia, py_inertia, rtol=1e-5, atol=1e-5)
