# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest
import hoomd
import hoomd.hpmc
from hoomd.hpmc._hpmc import PolyhedronVertices, EllipsoidParams
from hoomd.hpmc._hpmc import MassPropertiesConvexPolyhedron, MassPropertiesConvexSpheropolyhedron, MassPropertiesEllipsoid

shape_list = [
    # cube
    ("ConvexPolyhedron", {
        "vertices": [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1],
                     [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]],
        "sweep_radius": 0,
        "ignore_statistics": True
    }),
    # deformed cubed
    ("ConvexPolyhedron", {
        "vertices": [[-1.00010558, -1.05498028, -1.02785711],
                     [-0.60306277, -0.94357813, 1.15216173],
                     [-0.93980368, 1.10805945, 1.07469946],
                     [-0.92443732, 0.98195736, -1.46118865],
                     [1.228552, -1.30713086, -1.1727504],
                     [1.20330899, -0.79320722, 0.83510155],
                     [1.00378097, 0.92333129, 0.93916291],
                     [1.1994583, 0.97454523, -1.29517718]],
        "sweep_radius": 0,
        "ignore_statistics": True
    }),
    # truncated tetrahedron with truncation  = 0.4
    ("ConvexPolyhedron", {
        "vertices": [[-1., -0.6, 0.6], [-1., 0.6, -0.6], [-0.6, -1., 0.6],
                     [-0.6, -0.6, 1.], [-0.6, 0.6, -1.], [-0.6, 1., -0.6],
                     [0.6, -1., -0.6], [0.6, -0.6, -1.], [0.6, 0.6, 1.],
                     [0.6, 1., 0.6], [1., -0.6, -0.6], [1., 0.6, 0.6]],
        "sweep_radius": 0,
        "ignore_statistics": True
    }),
    # deformed truncated tetrahedron
    ("ConvexPolyhedron", {
        "vertices": [[-1.38818127, -0.43327022, 0.48655642],
                     [-0.76510261, 0.66381377, -0.56182591],
                     [-0.52614596, -1.02022957, 0.4116381],
                     [-0.88082834, -0.1838706, 0.97593675],
                     [-0.44804162, 0.96548643, -1.13214542],
                     [-0.76156125, 1.17756002, -0.64348949],
                     [0.4120951, -0.88009234, -0.15537746],
                     [0.80000109, -0.37006509, -1.03111526],
                     [0.26988483, 0.30897717, 1.06415232],
                     [0.76226993, 0.95183908, 0.63302422],
                     [0.99329989, -0.58242708, -0.39317069],
                     [0.78783469, 0.39728311, 0.51595356]],
        "sweep_radius": 0,
        "ignore_statistics": True
    }),
    ("Ellipsoid", {
        "a": 1,
        "b": 1,
        "c": 1,
        "ignore_statistics": True
    }),
    ("Ellipsoid", {
        "a": 1,
        "b": 1,
        "c": 2,
        "ignore_statistics": True
    }),
    ("Ellipsoid", {
        "a": 1.5,
        "b": 1,
        "c": 1,
        "ignore_statistics": True
    }),
    ("Ellipsoid", {
        "a": 1,
        "b": 0.8,
        "c": 1,
        "ignore_statistics": True
    }),
    ("Ellipsoid", {
        "a": 1.3,
        "b": 2.7,
        "c": 0.7,
        "ignore_statistics": True
    }),
    # sphere as convex spheropolyhedron
    ("ConvexSpheropolyhedron", {
        "vertices": [[0,0,0]],
        "sweep_radius": 1,
        "ignore_statistics": True
    }),
    # truncated tetrahedra (trunc = 0.1) as convex spheropolyhedron
    ("ConvexSpheropolyhedron", {
        "vertices": [[-1., -0.9, 0.9], [-1., 0.9, -0.9], [-0.9, -1., 0.9],
                     [-0.9, -0.9, 1.], [-0.9, 0.9, -1.], [-0.9, 1., -0.9],
                     [0.9, -1., -0.9], [0.9, -0.9, -1.], [0.9, 0.9, 1.],
                     [0.9, 1., 0.9], [1., -0.9, -0.9], [1., 0.9, 0.9]],
        "sweep_radius": 0,
        "ignore_statistics": True
    }),
]

volume_list = [
    8.0, 9.906613237811571, 2.5813333333333337, 3.406051603090999,
    4.1887902047863905, 8.377580409572781, 6.283185307179586,
    3.3510321638291125, 10.291857533160162, 4.1887902047863905, 2.6653333333333333
]

det_moi_list = [
    151.70370370370372, 433.6939933469258, 0.873927553653835, 2.754115599932528,
    4.70376701046326, 235.18835052316288, 41.920486071765346,
    1.6193602241717742, 1328.2618881466828, 4.70376701046326, 1.2054288640850612
]


def _get_cpp_cls(shape_type):
    if shape_type in ("ConvexPolyhedron", "ConvexSpheropolyhedron"):
        return (PolyhedronVertices, eval("MassProperties"+shape_type))
    elif shape_type == "Ellipsoid":
        return (EllipsoidParams, MassPropertiesEllipsoid)


@pytest.mark.parametrize("shape,volume", zip(shape_list, volume_list))
def test_volume(shape, volume):
    param_cpp_cls, mass_props_cpp_cls = _get_cpp_cls(shape[0])
    mass_props_obj = mass_props_cpp_cls(param_cpp_cls(shape[1]))
    assert np.allclose(mass_props_obj.getVolume(), volume)


@pytest.mark.parametrize("shape,det_moi", zip(shape_list, det_moi_list))
def test_det_moi(shape, det_moi):
    param_cpp_cls, mass_props_cpp_cls = _get_cpp_cls(shape[0])
    mass_props_obj = mass_props_cpp_cls(param_cpp_cls(shape[1]))
    assert np.allclose(mass_props_obj.getDetInertiaTensor(), det_moi)
