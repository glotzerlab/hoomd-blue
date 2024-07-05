# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest
import hoomd
from hoomd.hpmc.integrate import (ConvexPolygon, ConvexPolyhedron,
                                  ConvexSpheropolygon, Ellipsoid,
                                  FacetedEllipsoid, FacetedEllipsoidUnion,
                                  Polyhedron, SimplePolygon, Sphere,
                                  SphereUnion, ConvexSpheropolyhedron,
                                  ConvexSpheropolyhedronUnion, Sphinx)
from copy import deepcopy
from collections import Counter

_valid_args = [
    (ConvexPolygon, {
        'vertices': [(0, (0.75**0.5) / 2), (-0.5, -(0.75**0.5) / 2),
                     (0.5, -(0.75**0.5) / 2)]
    }, 2),
    (ConvexPolygon, {
        'vertices': [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    }, 2),
    (ConvexPolygon, {
        'vertices': [(-0.125, -0.125), (0.375, 0.125), (0.125, 0.375),
                     (-0.125, 0.125)],
        'sweep_radius': 0.3
    }, 2),
    (ConvexPolygon, {
        'vertices': [(0, 0), (0.25, 0), (0.5, 0.25), (0.25, 0.5), (0, 0.25)],
        'ignore_statistics': 1
    }, 2),
    (ConvexPolyhedron, {
        'vertices': [(0, (0.75**0.5) / 2, -0.5), (-0.5, -(0.75**0.5) / 2, -0.5),
                     (0.5, -(0.75**0.5) / 2, -0.5), (0, 0, 0.5)]
    }, 3),
    (ConvexPolyhedron, {
        'vertices': [(0, 0.25, 0), (0.375, 0.375, 0.375), (0.375, 0, 0.375),
                     (0, 0.375, 0.375), (0.375, 0.375, 0), (0, 0, 0.375)],
        'ignore_statistics': 1,
        'sweep_radius': 0.125
    }, 3),
    (ConvexPolyhedron, {
        'vertices': [(0.25, 0, 0), (0.25, 0.25, 0), (0.25, 0.5, 0.25),
                     (0, 0.25, 0.25), (0.25, 0.25, 0.5), (0, 0, 0.25)],
        'sweep_radius': 0.2
    }, 3),
    (ConvexPolyhedron, {
        'vertices': [(0, 0, 0), (0.25, 0.25, 0.25), (0.25, 0, 0.5),
                     (0.5, 0.25, 0.25)],
        'ignore_statistics': 1
    }, 3),
    (ConvexSpheropolygon, {
        'vertices': [(0, (0.75**0.5) / 2), (-0.5, -(0.75**0.5) / 2),
                     (0.5, -(0.75**0.5) / 2)]
    }, 2),
    (ConvexSpheropolygon, {
        'vertices': [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    }, 2),
    (ConvexSpheropolygon, {
        'vertices': [(-0.125, -0.125), (0.375, 0.125), (0.125, 0.375),
                     (-0.125, 0.125)],
        'sweep_radius': 0.3
    }, 2),
    (ConvexSpheropolygon, {
        'vertices': [(0, 0), (0.25, 0), (0.5, 0.25), (0.25, 0.5), (0, 0.25)],
        'ignore_statistics': 1
    }, 2),
    (ConvexSpheropolyhedron, {
        'vertices': [(0, (0.75**0.5) / 2, -0.5), (-0.5, -(0.75**0.5) / 2, -0.5),
                     (0.5, -(0.75**0.5) / 2, -0.5), (0, 0, 0.5)]
    }, 3),
    (ConvexSpheropolyhedron, {
        'vertices': [(0, 0.25, 0), (0.375, 0.375, 0.375), (0.375, 0, 0.375),
                     (0, 0.375, 0.375), (0.375, 0.375, 0), (0, 0, 0.375)],
        'ignore_statistics': 1,
        'sweep_radius': 0.125
    }, 3),
    (ConvexSpheropolyhedron, {
        'vertices': [(0.25, 0, 0), (0.25, 0.25, 0), (0.25, 0.5, 0.25),
                     (0, 0.25, 0.25), (0.25, 0.25, 0.5), (0, 0, 0.25)],
        'sweep_radius': 0.2
    }, 3),
    (ConvexSpheropolyhedron, {
        'vertices': [(0, 0, 0), (0.25, 0.25, 0.25), (0.25, 0, 0.5),
                     (0.5, 0.25, 0.25)],
        'ignore_statistics': 1
    }, 3),
    (Ellipsoid, {
        'a': 0.125,
        'b': 0.375,
        'c': 0.5
    }, 3),
    (Ellipsoid, {
        'a': 1.0 / 6.0,
        'b': 2.0 / 6.0,
        'c': 0.5
    }, 3),
    (Ellipsoid, {
        'a': 0.5,
        'b': 1.0 / 8.0,
        'c': 3.0 / 8.0,
        'ignore_statistics': 1
    }, 3),
    (Ellipsoid, {
        'a': 1.0 / 12.0,
        'b': 5.0 / 12.0,
        'c': 0.5,
        'ignore_statistics': 0
    }, 3),
    (FacetedEllipsoid, {
        "normals": [(0, 0, 1)],
        "a": 0.5,
        "b": 0.5,
        "c": 0.25,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": [0.125]
    }, 3),
    (FacetedEllipsoid, {
        "normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
        "offsets": [0.1, 0.25, 0.25],
        "a": 0.5,
        "b": 0.25,
        "c": 0.125,
        "vertices": [],
        "origin": (0, 0, 0)
    }, 3),
    (FacetedEllipsoid, {
        "normals": [(1, 0, 0)],
        "offsets": [0.25],
        "a": 0.5,
        "b": 0.25,
        "c": 0.5,
        "vertices": [],
        "origin": (0, 0, 0.125),
        "ignore_statistics": 1
    }, 3),
    (FacetedEllipsoid, {
        "normals": [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1),
                    (0, 0, 1)],
        "offsets": [-0.125, -1, -.5, -.5, -.5, -.5],
        "a": 0.5,
        "b": 0.5,
        "c": 0.5,
        "vertices": [[-.125, -.5, -.5], [-.125, -.5, .5], [-.125, .5, .5],
                     [-.125, .5, -.5], [1, -.5, -.5], [1, -.5, .5], [1, .5, .5],
                     [1, .5, -.5]],
        "origin": (0, 0.125, 0)
    }, 3),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }, 3),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': [(-0.1, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 0],
        'capacity': 3,
        'ignore_statistics': False
    }, 3),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': [(0.1, 0, 0.1), (0, 0, 0)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [0, 1]
    }, 3),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0)],
            "offsets": [0.1, 0.25, 0.25],
            "a": 0.5,
            "b": 0.25,
            "c": 0.125,
            "vertices": [],
            "origin": (0, 0, 0)
        }, {
            "normals": [(1, 0, 0)],
            "offsets": [0.25],
            "a": 0.5,
            "b": 0.25,
            "c": 0.5,
            "vertices": [],
            "origin": (0, 0, 0.125),
            "ignore_statistics": 1
        }],
        'positions': [(0, 0, 0), (0, 0, -0.1), (0.1, 0.1, 0.1)],
        'orientations': [(1, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1)],
        'overlap': [1, 1, 1],
        'capacity': 4,
        'ignore_statistics': 1
    }, 3),
    (Polyhedron, {
        "vertices": [(0.25, 0.25, 0.25), (-0.25, -0.25, 0.25),
                     (0.25, -0.25, -0.25), (-0.25, 0.25, -0.25)],
        "faces": [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]]
    }, 3),
    (Polyhedron, {
        'vertices': [(-0.25, -0.25, -0.25), (-0.25, -0.25, 0.25),
                     (-0.25, 0.25, -0.25), (-0.25, 0.25, 0.25),
                     (0.25, -0.25, -0.25), (0.25, -0.25, 0.25),
                     (0.25, 0.25, -0.25), (0.25, 0.25, 0.25)],
        'faces': [[0, 2, 6], [6, 4, 0], [5, 0, 4], [5, 1, 0], [5, 4, 6],
                  [5, 6, 7], [3, 2, 0], [3, 0, 1], [3, 6, 2], [3, 7, 6],
                  [3, 1, 5], [3, 5, 7]],
        'overlap': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'sweep_radius': 0.1
    }, 3),
    (Polyhedron, {
        'vertices': [(0, 0.3, 0), (0.2, 0.1, 0), (0.1, 0.3, 0.1),
                     (0.1, 0.1, 0.1), (0.1, 0.2, 0.5), (0.3, 0, 0.1),
                     (0, 0.3, 0.3)],
        'faces': [(0, 1, 2), (3, 2, 6), (1, 2, 4), (6, 1, 3), (3, 4, 6),
                  (4, 5, 1), (6, 2, 5)],
        'ignore_statistics': 1,
        'capacity': 4
    }, 3),
    (Polyhedron, {
        'vertices': [(0, 0.5, 0), (1 / 3, 1 / 6, 0), (0.5, 0, 1 / 6),
                     (0, 0.5, 0.5)],
        'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
        'capacity': 5,
        'hull_only': True
    }, 3),
    (SimplePolygon, {
        "vertices": [(0, (0.75**0.5) / 2), (0, 0), (-0.5, -(0.75**0.5) / 2),
                     (0.5, -(0.75**0.5) / 2)]
    }, 2),
    (SimplePolygon, {
        "vertices": [(-0.5, 0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, -0.5)]
    }, 2),
    (SimplePolygon, {
        "vertices": [(-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)],
        "ignore_statistics": 1
    }, 2),
    (SimplePolygon, {
        "vertices": [(-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)],
    }, 2),
    (Sphere, {
        "diameter": 1
    }, 3),
    (Sphere, {
        'diameter': 1.1,
        'ignore_statistics': 1
    }, 3),
    (Sphere, {
        'diameter': 0.9,
        'orientable': 1
    }, 3),
    (Sphere, {
        'diameter': 0.8,
        'orientable': 1,
        'ignore_statistics': 1
    }, 3),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }, 3),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 0.5
        }],
        'positions': [(0.2, 0, 0), (0, 0, 0.2)],
        'orientations': [(2**0.5, 2**0.5, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }, 3),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': [(0.2, 0.2, 0), (-0.1, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 0]
    }, 3),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 0.5
        }, {
            "diameter": 0.75
        }],
        'positions': [(0, 0, 0), (0, -0.1, -0.1), (0.1, 0.1, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1, 1],
        'capacity': 5,
        'ignore_statistics': 1
    }, 3),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(0.25, 0, 0), (0.25, 0.25, 0), (0.25, 0.5, 0.25),
                         (0, 0.25, 0.25), (0.25, 0.25, 0.5), (0, 0, 0.25)]
        }, {
            'vertices': [(0.25, 0, 0), (0.25, 0.25, 0), (0.25, 0.5, 0.25),
                         (0, 0.25, 0.25), (0.25, 0.25, 0.5), (0, 0, 0.25)]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }, 3),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(0.25, 0, 0), (0.25, 0.25, 0), (0.25, 0.5, 0.25),
                         (0, 0.25, 0.25), (0.25, 0.25, 0.5), (0, 0, 0.25)]
        }, {
            'vertices': [(0, 0, 0), (0.25, 0.25, 0.25), (0.25, 0, 0.5),
                         (0.5, 0.25, 0.25)]
        }],
        'positions': [(-0.1, 0, 0), (0, 0, 0.1)],
        'orientations': [(2**0.5, 2**0.5, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }, 3),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(0, 0, 0), (0.25, 0.25, 0.25), (0.25, 0, 0.5),
                         (0.5, 0.25, 0.25)]
        }, {
            'vertices': [(0, 0, 0), (0.25, 0.25, 0.25), (0.25, 0, 0.5),
                         (0.5, 0.25, 0.25)]
        }],
        'positions': [(-0.1, -0.1, 0), (0.1, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 0]
    }, 3),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(0, 0, 0), (0.25, 0.25, 0.25), (0.25, 0, 0.5),
                         (0.5, 0.25, 0.25)]
        }, {
            'vertices': [(0.25, 0, 0), (0.25, 0.25, 0), (0.25, 0.5, 0.25),
                         (0, 0.25, 0.25), (0.25, 0.25, 0.5), (0, 0, 0.25)]
        }, {
            'vertices': [(0, 0, 0), (0.25, 0.25, 0.25), (0.25, 0, 0.5),
                         (0.5, 0.25, 0.25)]
        }],
        'positions': [(0, 0, 0), (0, -0.1, -0.1), (0.1, 0.1, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1, 1],
        'capacity': 5,
        'ignore_statistics': 1
    }, 3),
    (Sphinx, {
        'diameters': [1, -.001],
        'centers': [(0, 0, 0), (0.5, 0, 0)]
    }, 3),
    (Sphinx, {
        'diameters': [1, -1],
        'centers': [(0, 0, 0), (0.75, 0, 0)]
    }, 3),
    (Sphinx, {
        'diameters': [1, -0.5],
        'centers': [(0, 0, 0), (0, 0, .6)]
    }, 3),
    (Sphinx, {
        'diameters': [1, -0.25],
        'centers': [(0, 0, 0), (0.6, 0, 0)],
        'ignore_statistics': 1
    }, 3),
]

_invalid_args = [
    (ConvexPolygon, {
        'vertices': "str"
    }),
    (ConvexPolygon, {
        'vertices': 1
    }),
    (ConvexPolygon, {
        'vertices': [(0, 0), (1, 1), (1, 0), (0, 1), (1, 1), (0, 0), (2, 1),
                     (1, 3)],
        'sweep_radius': "str"
    }),
    (ConvexPolyhedron, {
        'vertices': "str"
    }),
    (ConvexPolyhedron, {
        'vertices': 1
    }),
    (ConvexPolyhedron, {
        'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)],
        'sweep_radius': "str"
    }),
    (ConvexSpheropolygon, {
        'vertices': "str"
    }),
    (ConvexSpheropolygon, {
        'vertices': 1
    }),
    (ConvexSpheropolygon, {
        'vertices': [(0, 0), (1, 1), (1, 0), (0, 1), (1, 1), (0, 0), (2, 1),
                     (1, 3)],
        'sweep_radius': "str"
    }),
    (ConvexSpheropolyhedron, {
        'vertices': "str"
    }),
    (ConvexSpheropolyhedron, {
        'vertices': 1
    }),
    (ConvexSpheropolyhedron, {
        'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)],
        'sweep_radius': "str"
    }),
    (Ellipsoid, {
        'a': 'str',
        'b': 'str',
        'c': 'str'
    }),
    (Ellipsoid, {
        'a': 1,
        'b': 3,
        'c': 'str'
    }),
    (Ellipsoid, {
        'a': [1, 2, 3],
        'b': [3, 7, 7],
        'c': [2, 5, 9]
    }),
    (FacetedEllipsoid, {
        "normals": "str",
        "a": 1,
        "b": 1,
        "c": 0.5,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": [0]
    }),
    (FacetedEllipsoid, {
        "normals": [(0, 0, 1)],
        "a": "str",
        "b": 1,
        "c": 0.5,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": [0]
    }),
    (FacetedEllipsoid, {
        "normals": [(0, 0, 1)],
        "a": 1,
        "b": 1,
        "c": 0.5,
        "vertices": "str",
        "origin": (0, 0, 0),
        "offsets": [0]
    }),
    (FacetedEllipsoid, {
        "normals": [(0, 0, 1)],
        "a": 1,
        "b": 1,
        "c": 0.5,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": "str"
    }),
    (FacetedEllipsoid, {
        "normals": 1,
        "a": 1,
        "b": 1,
        "c": 0.5,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": [0]
    }),
    (FacetedEllipsoid, {
        "normals": [(0, 0, 1)],
        "a": [1, 2, 3],
        "b": 1,
        "c": 0.5,
        "vertices": [],
        "origin": (0, 0, 0),
        "offsets": [0]
    }),
    (FacetedEllipsoid, {
        "normals": [(0, 0, 1)],
        "a": 1,
        "b": 1,
        "c": 0.5,
        "vertices": 4,
        "origin": (0, 0, 0),
        "offsets": [0]
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': "str",
        'positions': [(1, 0, 0), (0, 0, 1)],
        'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 0]
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': "str",
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [0, 1]
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': "str",
        'overlap': [1, 1]
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': "str"
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': 1,
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': 1,
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': 1,
        'overlap': [1, 1]
    }),
    ((FacetedEllipsoid, FacetedEllipsoidUnion), {
        'shapes': [{
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }, {
            "normals": [(0, 0, 1)],
            "a": 0.5,
            "b": 0.5,
            "c": 0.25,
            "vertices": [],
            "origin": (0, 0, 0),
            "offsets": [0.125]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': 1
    }),
    (Polyhedron, {
        "vertices": "str",
        "faces": [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]]
    }),
    (Polyhedron, {
        "vertices": [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, -0.5),
                     (-0.5, 0.5, -0.5)],
        "faces": "str"
    }),
    (Polyhedron, {
        "vertices": 1,
        "faces": [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]]
    }),
    (Polyhedron, {
        "vertices": [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, -0.5),
                     (-0.5, 0.5, -0.5)],
        "faces": 1
    }),
    (Polyhedron, {
        "vertices": [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, -0.5),
                     (-0.5, 0.5, -0.5)],
        "faces": [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]],
        'overlap': "str"
    }),
    (Polyhedron, {
        "vertices": [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, -0.5),
                     (-0.5, 0.5, -0.5)],
        "faces": [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]],
        'overlap': 1
    }),
    (Polyhedron, {
        "vertices": [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, -0.5),
                     (-0.5, 0.5, -0.5)],
        "faces": [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]],
        'sweep_radius': "str"
    }),
    (Polyhedron, {
        "vertices": [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, -0.5),
                     (-0.5, 0.5, -0.5)],
        "faces": [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]],
        'capacity': "str"
    }),
    (SimplePolygon, {
        "vertices": "str"
    }),
    (SimplePolygon, {
        "vertices": 1
    }),
    (SimplePolygon, {
        "vertices": [(-1, 1), (1, -1), (1, 1)],
        "sweep_radius": "str"
    }),
    (Sphere, {
        "diameter": "str"
    }),
    (Sphere, {
        'diameter': [1, 2, 3]
    }),
    ((Sphere, SphereUnion), {
        'shapes': "str",
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': "str",
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': "str",
        'overlap': [1, 1]
    }),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': "str"
    }),
    ((Sphere, SphereUnion), {
        'shapes': 1,
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': 1,
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': 1,
        'overlap': [1, 1]
    }),
    ((Sphere, SphereUnion), {
        'shapes': [{
            "diameter": 1
        }, {
            "diameter": 1
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': 1
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': "str",
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }, {
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }],
        'positions': "str",
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }, {
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': "str",
        'overlap': [1, 1]
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }, {
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': "str"
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': 1,
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }, {
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }],
        'positions': 1,
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': [1, 1]
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }, {
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': 1,
        'overlap': [1, 1]
    }),
    ((ConvexSpheropolyhedron, ConvexSpheropolyhedronUnion), {
        'shapes': [{
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)]
        }, {
            'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2),
                         (0, 0, 1)],
            'sweep_radius': 0.3
        }],
        'positions': [(0, 0, 0), (0, 0, 0.1)],
        'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
        'overlap': 1
    }),
    (Sphinx, {
        'diameters': "str",
        'centers': [(0, 0, 0), (0.8, 0, 0)]
    }),
    (Sphinx, {
        'diameters': [1, -1],
        'centers': "str"
    }),
    (Sphinx, {
        'diameters': 1,
        'centers': [(0, 0, 0), (0, 0, .6)]
    }),
    (Sphinx, {
        'diameters': [0.5, -0.25],
        'centers': 1
    }),
]


class CounterWrapper:

    def __init__(self, func):
        self.func = func
        self._counter = []

    def __call__(self, *args, **kwargs):
        self._counter.append(str(args[0][0]).split('.')[-1][:-2])
        return self.func(*args, **kwargs)

    def count(self, integrator):
        return Counter(self._counter)[str(integrator).split('.')[-1][:-2]]


@CounterWrapper
def valid_args_id(args):
    integrator = args[0]
    if isinstance(integrator, tuple):
        name = integrator[1].__name__
    else:
        name = integrator.__name__

    return name + '-' + str(valid_args_id.count(str(integrator)))


@pytest.fixture(scope="function", params=_valid_args, ids=valid_args_id)
def valid_args(request):
    return deepcopy(request.param)


@CounterWrapper
def invalid_args_id(args):
    integrator = args[0]
    if isinstance(integrator, tuple):
        name = integrator[1].__name__
    else:
        name = integrator.__name__
    return name + '-' + str(valid_args_id.count(str(integrator)))


@pytest.fixture(scope="function", params=_invalid_args, ids=invalid_args_id)
def invalid_args(request):
    return deepcopy(request.param)


def _test_moves_id(args):
    integrator = args[0]
    return integrator.__name__


def _test_moves_args(_valid_args):
    integrator_str = []
    args_list = []
    for integrator, args, n_dimensions in _valid_args:
        if isinstance(integrator, tuple):
            integrator = integrator[1]
        if str(integrator) not in integrator_str:
            integrator_str.append(str(integrator))
            args_list.append((integrator, args, n_dimensions))
    return args_list


@pytest.fixture(scope="function",
                params=_test_moves_args(_valid_args),
                ids=_test_moves_id)
def test_moves_args(request):
    return deepcopy(request.param)


def _cpp_args(_valid_args):
    args_list = []
    for integrator, args, n_dimensions in _valid_args:
        cpp_shape = None
        if 'SphereUnion' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.SphereUnionParams
        elif 'Sphere' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.SphereParams
        elif 'Sphinx' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.SphinxParams
        elif 'FacetedEllipsoidUnion' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.mfellipsoid_params
        elif 'FacetedEllipsoid' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.FacetedEllipsoidParams
        elif 'Ellipsoid' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.EllipsoidParams
        elif 'ConvexSpheropolyhedronUnion' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.mpoly3d_params
        elif 'polygon' in str(integrator).lower():
            cpp_shape = hoomd.hpmc._hpmc.PolygonVertices
        elif 'Convex' in str(integrator):
            cpp_shape = hoomd.hpmc._hpmc.PolyhedronVertices
        if cpp_shape:
            if isinstance(integrator, tuple):
                inner_integrator = integrator[0]
                integrator = integrator[1]
                inner_mc = inner_integrator()
                for i in range(len(args["shapes"])):
                    # This will fill in default values for the inner shape
                    # objects
                    inner_mc.shape["A"] = args["shapes"][i]
                    args["shapes"][i] = inner_mc.shape["A"].to_base()
            mc = integrator()
            mc.shape['A'] = args
            args_list.append((cpp_shape, mc.shape['A'].to_base()))
    return args_list


@CounterWrapper
def cpp_args_id(args):
    integrator = args[0]
    return str(integrator) + str(valid_args_id.count(str(integrator)))


@pytest.fixture(scope="function",
                params=_cpp_args(_valid_args),
                ids=cpp_args_id)
def cpp_args(request):
    return deepcopy(request.param)
