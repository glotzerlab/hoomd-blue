import pytest
import hoomd
import numpy as np


@pytest.fixture(scope="function")
def convex_polygon_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.ConvexPolygon
    return make_shape


@pytest.fixture(scope="function")
def convex_polygon_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.PolygonVertices
    return get_parameters


@pytest.fixture(scope="function")
def convex_polygon_valid_args():
    def get_args():
        args_list = [{'vertices': [(0, (0.75**0.5) / 2),
                                   (-0.5, -(0.75**0.5) / 2),
                                   (0.5, -(0.75**0.5) / 2)],
                      'ignore_statistics': 0},
                     {'vertices': [(-0.5, -0.5),
                                   (0.5, -0.5),
                                   (0.5, 0.5),
                                   (-0.5, 0.5)]},
                     {'vertices': [(-0.5, -0.5),
                                   (4.5, 0.5),
                                   (0.5, 2.5),
                                   (-0.5, 0.5)],
                      'ignore_statistics': 0,
                      'sweep_radius': 0.3},
                     {'vertices': [(0, 0), (1, 0), (2, 1), (1, 3),
                                   (0, 1)],
                      'ignore_statistics': 1,
                      'sweep_radius': 0}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_polygon_invalid_args():
    def get_args():
        args_list = [{'vertices': "str"},
                     {'ignore_statistics': 1},
                     {'vertices': 1},
                     {'vertices': [(0, 0), (1, 1), (1, 0), (0, 1),
                                   (1, 1), (0, 0), (2, 1), (1, 3)],
                      'sweep_radius': "str"},
                     {'vertices': [(0, 0), (1, 1), (1, 0), (0, 1),
                                   (1, 1), (0, 0), (2, 1), (1, 3)],
                      'ignore_statistics': "str"}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_polyhedron_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.ConvexPolyhedron
    return make_shape


@pytest.fixture(scope="function")
def convex_polyhedron_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.PolyhedronVertices
    return get_parameters


@pytest.fixture(scope="function")
def convex_polyhedron_valid_args():
    def get_args():
        args_list = [{'vertices': [(0, (0.75**0.5) / 2, -0.5),
                                   (-0.5, -(0.75**0.5) / 2, -0.5),
                                   (0.5, -(0.75**0.5) / 2, -0.5),
                                   (0, 0, 0.5)]},
                     {'vertices': [(0, 5, 0), (1, 1, 1), (1, 0, 1),
                                   (0, 1, 1), (1, 1, 0), (0, 0, 1)],
                      'ignore_statistics': 1,
                      'sweep_radius': 2.0},
                     {'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                      'sweep_radius': 1.0},
                     {'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2),
                                   (2, 1, 1)],
                      'ignore_statistics': 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_polyhedron_invalid_args():
    def get_args():
        args_list = [{'vertices': "str"},
                     {'ignore_statistics': 1},
                     {'vertices': 1},
                     {'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2),
                                   (2, 1, 1)],
                      'sweep_radius': "str"},
                     {'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2),
                                   (2, 1, 1)],
                      'ignore_statistics': "str"}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_spheropolygon_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.ConvexSpheropolygon
    return make_shape


@pytest.fixture(scope="function")
def convex_spheropolygon_valid_args(convex_polygon_valid_args):
    def get_args():
        return convex_polygon_valid_args()
    return get_args


@pytest.fixture(scope="function")
def convex_spheropolygon_invalid_args(convex_polygon_invalid_args):
    def get_args():
        return convex_polygon_invalid_args()
    return get_args


@pytest.fixture(scope="function")
def convex_spheropolyhedron_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.ConvexSpheropolyhedron
    return make_shape


@pytest.fixture(scope="function")
def convex_spheropolyhedron_valid_args(convex_polyhedron_valid_args):
    def get_args():
        return convex_polyhedron_valid_args()
    return get_args


@pytest.fixture(scope="function")
def convex_spheropolyhedron_invalid_args(convex_polyhedron_invalid_args):
    def get_args():
        return convex_polyhedron_invalid_args()
    return get_args


@pytest.fixture(scope="function")
def ellipsoid_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.Ellipsoid
    return make_shape


@pytest.fixture(scope="function")
def ellipsoid_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.EllipsoidParams
    return get_parameters


@pytest.fixture(scope="function")
def ellipsoid_valid_args():
    def get_args():
        args_list = [{'a': 0.75, 'b': 1, 'c': 0.5},
                     {'a': 1, 'b': 2, 'c': 3},
                     {'a': 4, 'b': 1, 'c': 30, 'ignore_statistics': 1},
                     {'a': 10, 'b': 5, 'c': 6, 'ignore_statistics': 0}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def ellipsoid_invalid_args():
    def get_args():
        args_list = [{'a': 'str', 'b': 'str', 'c': 'str'},
                     {'a': 1, 'b': 3, 'c': 'str'},
                     {'a': [1, 2, 3], 'b': [3, 7, 7], 'c': [2, 5, 9]},
                     {'a': 'str', 'b': 'str', 'c': [1, 2, 3]},
                     {'ignore_statistics': 1},
                     {'a': 1, 'b': 0.5, 'c': 0.25, 'ignore_statistics': 'str'}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def faceted_ellipsoid_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.FacetedEllipsoid
    return make_shape


@pytest.fixture(scope="function")
def faceted_ellipsoid_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.FacetedEllipsoidParams
    return get_parameters


@pytest.fixture(scope="function")
def faceted_ellipsoid_valid_args():
    def get_args():
        args_list = [{"normals": [(0, 0, 1)],
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "origin": (0, 0, 0),
                      "offsets": [0]},
                     {"normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0),
                                  (0, 1, 1), (1, 1, 0), (1, 0, 1)],
                      "offsets": [1, 3, 2, 6, 3, 1],
                      "a": 3,
                      "b": 4,
                      "c": 1,
                      "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0),
                                   (1, 0, 0), (1, 1, 1), (1, 1, 0)],
                      "origin": (0, 0, 0)},
                     {"normals": [(0, 0, 0), (2, 1, 1), (1, 3, 3),
                                  (5, 1, 1), (1, 3, 0), (1, 2, 2)],
                      "offsets": [1, 3, 3, 2, 3, 1],
                      "a": 2,
                      "b": 1,
                      "c": 3,
                      "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                      "origin": (0, 0, 1),
                      "ignore_statistics": 1},
                     {"normals": [(0, 0, 2), (0, 1, 1),
                                  (1, 3, 5), (0, 1, 6)],
                      "offsets": [6, 2, 2, 5],
                      "a": 1,
                      "b": 6,
                      "c": 6,
                      "vertices": [(0, 0, 0), (1, 1, 1),
                                   (1, 0, 2), (2, 1, 1)],
                      "origin": (0, 1, 0)}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def faceted_ellipsoid_invalid_args():
    def get_args():
        args_list = [{"normals": "str",
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "origin": (0, 0, 0),
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "a": "str",
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "origin": (0, 0, 0),
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": "str",
                      "origin": (0, 0, 0),
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "origin": "str",
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "origin": (0, 0, 0),
                      "offsets": "str"},
                     {"a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "origin": (0, 0, 0),
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "vertices": [],
                      "origin": (0, 0, 0),
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "origin": (0, 0, 0),
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "offsets": [0]},
                     {"normals": [(0, 0, 1)],
                      "a": 1,
                      "b": 1,
                      "c": 0.5,
                      "vertices": [],
                      "origin": (0, 0, 0)}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def faceted_ellipsoid_union_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.FacetedEllipsoidUnion
    return make_shape


@pytest.fixture(scope="function")
def faceted_ellipsoid_union_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.mfellipsoid_params
    return get_parameters


@pytest.fixture(scope="function")
def faceted_ellipsoid_union_valid_args(faceted_ellipsoid_valid_args):
    def get_args():
        faceted_ell_args_list = faceted_ellipsoid_valid_args()
        args_list = [{'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                      'overlap': [0, 1]},
                     {'shapes': [faceted_ell_args_list[2],
                                 faceted_ell_args_list[1]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 0],
                      'capacity': 3},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'ignore_statistics': 1},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[1],
                                 faceted_ell_args_list[2]],
                      'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)],
                      'orientations': [(1, 1, 1, 1),
                                       (1, 0, 0, 0),
                                       (1, 0, 0, 1)],
                      'overlap': [1, 1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def faceted_ellipsoid_union_invalid_args(faceted_ellipsoid_valid_args):
    def get_args():
        faceted_ell_args_list = faceted_ellipsoid_valid_args()
        args_list = [{'shapes': "str",
                      'positions': [(1, 0, 1), (0, 0, 0)]},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[1]],
                      'positions': "str"},
                     {'shapes': [faceted_ell_args_list[2],
                                 faceted_ell_args_list[1]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': "str"},
                     {'shapes': [faceted_ell_args_list[1],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 1, 0), (0, 0, 1)],
                      'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)],
                      'overlap': "str"},
                     {'shapes': [faceted_ell_args_list[1],
                                 faceted_ell_args_list[0]],
                      'positions': 1},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'orientations': 2},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'capacity': "str"},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]]},
                     {'positions': [(1, 0, 1), (0, 0, 0)]}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def polyhedron_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.Polyhedron
    return make_shape


@pytest.fixture(scope="function")
def polyhedron_valid_args():
    def get_args():
        args_list = [{"vertices": [(0, (0.75**0.5) / 2, -0.5),
                                   (-0.5, -(0.75**0.5) / 2, -0.5),
                                   (0.5, -(0.75**0.5) / 2, -0.5),
                                   (0, 0, 0.5),
                                   (0, 0, 0)],
                      "faces": [(0, 3, 1),
                                (0, 2, 3),
                                (1, 3, 2),
                                (1, 2, 4),
                                (0, 1, 4),
                                (0, 4, 2)],
                      "overlap": [1, 0, 1, 1, 0, 0]},
                     {'vertices': [(0, 0, 0),
                                   (1, 1, 0),
                                   (1, 0, 1),
                                   (0, 1, 1),
                                   (1, 1, 1),
                                   (0, 0, 1)],
                      'faces': [(0, 0, 0),
                                (1, 1, 0),
                                (1, 0, 1),
                                (0, 1, 1),
                                (1, 1, 1),
                                (0, 0, 1)],
                      'overlap': [1, 1, 1, 1, 1, 1],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': 4,
                      'origin': (0, 0, 0),
                      'hull_only': True},
                     {'vertices': [(0, 3, 0),
                                   (2, 1, 0),
                                   (1, 3, 1),
                                   (1, 1, 1),
                                   (1, 2, 5),
                                   (3, 0, 1)],
                      'faces': [(0, 4, 5),
                                (1, 3, 2),
                                (1, 2, 5),
                                (5, 1, 3),
                                (1, 4, 3),
                                (0, 2, 1)],
                      'overlap': [0, 0, 0, 0, 0, 0],
                      'sweep_radius': 2},
                     {'vertices': [(0, 3, 0),
                                   (2, 1, 0),
                                   (1, 3, 1),
                                   (1, 1, 1),
                                   (1, 2, 5),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2),
                                (3, 2, 6),
                                (1, 2, 4),
                                (6, 1, 3),
                                (3, 4, 6),
                                (4, 5, 1),
                                (6, 2, 5)],
                      'ignore_statistics': 1,
                      'capacity': 4,
                      'origin': (0, 0, 0),
                      'hull_only': True},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': [False, False, False, False],
                      'hull_only': True},
                     {'vertices': [(0, 3, 0),
                                   (2, 1, 0),
                                   (1, 3, 1),
                                   (1, 1, 1),
                                   (1, 2, 5),
                                   (3, 0, 1),
                                   (0, 3, 3),
                                   (0, 0, 2),
                                   (1, 2, 2)],
                      'faces': [(0, 1, 2),
                                (3, 2, 6),
                                (1, 2, 4),
                                (6, 1, 3),
                                (3, 4, 6),
                                (4, 5, 1),
                                (6, 7, 5),
                                (1, 7, 8),
                                (6, 8, 2)]}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def polyhedron_invalid_args():
    def get_args():
        args_list = [{'vertices': "str",
                      'faces': [(0, 0, 0),
                                (1, 1, 0),
                                (1, 0, 1),
                                (0, 1, 1),
                                (1, 1, 1),
                                (0, 0, 1)]},
                     {'vertices': [(0, 3, 0),
                                   (2, 1, 0),
                                   (1, 3, 1),
                                   (1, 1, 1),
                                   (1, 2, 5),
                                   (3, 0, 1)],
                      'faces': "str"},
                     {'vertices': [(0, 3, 0),
                                   (2, 1, 0),
                                   (1, 3, 1),
                                   (1, 1, 1),
                                   (1, 2, 5),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2),
                                (3, 2, 6),
                                (1, 2, 4),
                                (6, 1, 3),
                                (3, 4, 6),
                                (4, 5, 1),
                                (6, 2, 5)],
                      'overlap': "str"},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'sweep_radius': "str"},
                     {'vertices': [(0, 3, 0),
                                   (2, 1, 0),
                                   (1, 3, 1),
                                   (1, 1, 1),
                                   (1, 2, 5),
                                   (3, 0, 1),
                                   (0, 3, 3),
                                   (0, 0, 2),
                                   (1, 2, 2)],
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'ignore_statistics': "str"},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'capacity': "str"},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'origin': "str"},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'hull_only': "str"},
                     {'vertices': 1,
                      'faces': [(0, 1, 2),
                                (3, 2, 6),
                                (1, 2, 4),
                                (6, 1, 3),
                                (3, 4, 6),
                                (4, 5, 1),
                                (6, 7, 5),
                                (1, 7, 8),
                                (6, 8, 2)]},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': 1},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)],
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': 1},
                     {'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)]},
                     {'vertices': [(0, 3, 0), 
                                   (2, 1, 0),
                                   (3, 0, 1),
                                   (0, 3, 3)]}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def simple_polygon_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.SimplePolygon
    return make_shape


@pytest.fixture(scope="function")
def simple_polygon_valid_args():
    def get_args():
        args_list = [{"vertices": [(0, (0.75**0.5) / 2),
                                   (0, 0),
                                   (-0.5, -(0.75**0.5) / 2),
                                   (0.5, -(0.75**0.5) / 2)]},
                     {"vertices": [(-1, 1), (1, -1), (1, 1), (-1, -1)]},
                     {"vertices": [(-1, 1), (1, -1), (1, 1)],
                      "ignore_statistics": 1},
                     {"vertices": [(-1, 1), (1, -1), (1, 1)],
                      "sweep_radius": 2}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def simple_polygon_invalid_args():
    def get_args():
        args_list = [{"vertices": "str"},
                     {"vertices": 1},
                     {"ignore_statistics": 1},
                     # {"vertices": [(-1, 1), (1, -1), (1, 1)],
                     #  "ignore_statistics": "str"},
                     {"vertices": [(-1, 1), (1, -1), (1, 1)],
                      "sweep_radius": "str"}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphere_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.Sphere
    return make_shape


@pytest.fixture(scope="function")
def sphere_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.SphereParams
    return get_parameters


@pytest.fixture(scope="function")
def sphere_valid_args():
    def get_args():
        args_list = [{"diameter": 1,},
                     {'diameter': 1, 'ignore_statistics': 1},
                     {'diameter': 9, 'orientable': 1},
                     {'diameter': 4, 'orientable': 1, 'ignore_statistics': 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphere_invalid_args():
    def get_args():
        args_list = [{"diameter": "str"},
                     {"diameter": [1, 2, 3, 4]},
                     {"ignore_statistics": 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphere_union_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.SphereUnion
    return make_shape


@pytest.fixture(scope="function")
def sphere_union_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.SphereUnionParams
    return get_parameters


@pytest.fixture(scope="function")
def sphere_union_valid_args(sphere_valid_args):
    def get_args():
        sphere_args_list = sphere_valid_args()
        args_list = [{'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': [sphere_args_list[0], sphere_args_list[2]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': [(2**0.5, 2**0.5, 0, 0), (1, 0, 0, 0)]},
                     {'shapes': [sphere_args_list[2], sphere_args_list[1]],
                      'positions': [(1, 1, 0), (1, 0, 1)],
                      'overlap': [1, 0]},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1],
                                 sphere_args_list[2]],
                      'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1)],
                      'capacity': 4,
                      'ignore_statistics': 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphere_union_invalid_args(sphere_valid_args):
    def get_args():
        sphere_args_list = sphere_valid_args()
        args_list = [{'shapes': 'str',
                      'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': 'str'},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'orientations': 'str'},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'overlap': 'str'},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'capacity': 'str'},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]]},
                     {'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': 1,
                      'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': 1},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'orientations': 2},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'overlap': 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_spheropolyhedron_union_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion
    return make_shape


@pytest.fixture(scope="function")
def convex_spheropolyhedron_union_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.mpoly3d_params
    return get_parameters


@pytest.fixture(scope="function")
def convex_spheropolyhedron_union_valid_args(convex_polyhedron_valid_args):
    def get_args():
        polyhedron_vertices_args_list = convex_polyhedron_valid_args()
        args_list = [{'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': [polyhedron_vertices_args_list[2],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': [(2**0.5, 2**0.5, 0, 0), (1, 0, 0, 0)]},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'overlap': [0, 1],
                      'ignore_statistics': 1},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1],
                                 polyhedron_vertices_args_list[2]],
                      'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)],
                      'capacity': 4}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_spheropolyhedron_union_invalid_args(convex_polyhedron_valid_args):
    def get_args():
        polyhedron_vertices_args_list = convex_polyhedron_valid_args()
        args_list = [{'shapes': 'str',
                      'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': 'str'},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'orientations': 'str'},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'overlap': 'str'},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'capacity': 'str'},
                     {'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]]},
                     {'shapes': 1,
                      'positions': [(0, 0, 0), (0, 0, 1)]},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': 1},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'orientations': 2},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'overlap': 2}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphinx_integrator():
    def make_shape():
        return hoomd.hpmc.integrate.Sphinx
    return make_shape


@pytest.fixture(scope="function")
def sphinx_cpp():
    def get_parameters():
        return hoomd.hpmc._hpmc.SphinxParams
    return get_parameters


@pytest.fixture(scope="function")
def sphinx_valid_args():
    def get_args():
        args_list = [{'diameters': [1.6, -.001],
                      'centers': [(0, 0, 0), (0.5, 0, 0)]},
                     {'diameters': [1, 4, 2, 8, 5, 9],
                      'centers': [(0, 0, 0),
                                  (1, 1, 1),
                                  (1, 0, 1),
                                  (0, 1, 1),
                                  (1, 1, 0),
                                  (0, 0, 1)],
                      'ignore_statistics': 1},
                     {'diameters': [5, 2, 4, 5, 1, 2],
                      'centers': [(0, 2, 0),
                                  (1, 4, 1),
                                  (3, 0, 1),
                                  (3, 1, 1),
                                  (1, 4, 0),
                                  (2, 2, 1)],
                      'ignore_statistics': 0},
                     {'diameters': [1, 2, 2, 3, 4, 9, 3, 2],
                      'centers': [(0, 0, 0),
                                  (1, 1, 1),
                                  (1, 0, 1),
                                  (0, 1, 1),
                                  (1, 1, 0),
                                  (0, 0, 1),
                                  (2, 2, 1),
                                  (3, 5, 3)]}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphinx_invalid_args():
    def get_args():
        args_list = [{'diameters': 'str',
                      'centers': [(0, 0, 0), (1, 1, 1)]},
                     {'diameters': [1, -0.1],
                      'centers': 'str'},
                     {'diameters': [1, -0.1]},
                     {'centers': [(0, 0, 0), (1, 1, 1)]},
                     {'diameters': 2,
                      'centers': [(0, 0, 0), (1, 1, 1)]},
                     {'diameters': [1, -0.1],
                      'centers': 2}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def shape_dict_conversion_args(convex_polygon_cpp,
                               convex_polygon_valid_args,
                               convex_polyhedron_cpp,
                               convex_polyhedron_valid_args,
                               ellipsoid_cpp,
                               ellipsoid_valid_args,
                               faceted_ellipsoid_cpp,
                               faceted_ellipsoid_valid_args,
                               faceted_ellipsoid_union_cpp,
                               faceted_ellipsoid_union_valid_args,
                               sphere_cpp,
                               sphere_valid_args,
                               sphere_union_cpp,
                               sphere_union_valid_args,
                               convex_spheropolyhedron_union_cpp,
                               convex_spheropolyhedron_union_valid_args,
                               sphinx_cpp,
                               sphinx_valid_args):
    def get_valid_args():
        return [(convex_polygon_cpp(),
                 convex_polygon_valid_args()),
                (convex_polyhedron_cpp(),
                 convex_polyhedron_valid_args()),
                (ellipsoid_cpp(),
                 ellipsoid_valid_args()),
                (faceted_ellipsoid_cpp(),
                 faceted_ellipsoid_valid_args()),
                (faceted_ellipsoid_union_cpp(),
                 faceted_ellipsoid_union_valid_args()),
                (sphere_cpp(),
                 sphere_valid_args()),
                (sphere_union_cpp(),
                 sphere_union_valid_args()),
                (convex_spheropolyhedron_union_cpp(),
                 convex_spheropolyhedron_union_valid_args()),
                (sphinx_cpp(),
                 sphinx_valid_args())]
    return get_valid_args


@pytest.fixture(scope="function")
def integrator_args(convex_polygon_integrator,
                    convex_polygon_valid_args,
                    convex_polygon_invalid_args,
                    convex_polyhedron_integrator,
                    convex_polyhedron_valid_args,
                    convex_polyhedron_invalid_args,
                    convex_spheropolygon_integrator,
                    convex_spheropolygon_valid_args,
                    convex_spheropolygon_invalid_args,
                    convex_spheropolyhedron_integrator,
                    convex_spheropolyhedron_valid_args,
                    convex_spheropolyhedron_invalid_args,
                    ellipsoid_integrator,
                    ellipsoid_valid_args,
                    ellipsoid_invalid_args,
                    faceted_ellipsoid_integrator,
                    faceted_ellipsoid_valid_args,
                    faceted_ellipsoid_invalid_args,
                    faceted_ellipsoid_union_integrator,
                    faceted_ellipsoid_union_valid_args,
                    faceted_ellipsoid_union_invalid_args,
                    polyhedron_integrator,
                    polyhedron_valid_args,
                    polyhedron_invalid_args,
                    simple_polygon_integrator,
                    simple_polygon_valid_args,
                    simple_polygon_invalid_args,
                    sphere_integrator,
                    sphere_valid_args,
                    sphere_invalid_args,
                    sphere_union_integrator,
                    sphere_union_valid_args,
                    sphere_union_invalid_args,
                    convex_spheropolyhedron_union_integrator,
                    convex_spheropolyhedron_union_valid_args,
                    convex_spheropolyhedron_union_invalid_args,
                    sphinx_integrator,
                    sphinx_valid_args,
                    sphinx_invalid_args):
    def get_args():
        return [(convex_polygon_integrator(),
                 convex_polygon_valid_args(),
                 convex_polygon_invalid_args()),
                (convex_polyhedron_integrator(),
                 convex_polyhedron_valid_args(),
                 convex_polyhedron_invalid_args()),
                (convex_spheropolygon_integrator(),
                 convex_spheropolygon_valid_args(),
                 convex_spheropolygon_invalid_args()),
                (convex_spheropolyhedron_integrator(),
                 convex_spheropolyhedron_valid_args(),
                 convex_spheropolyhedron_invalid_args()),
                (ellipsoid_integrator(),
                 ellipsoid_valid_args(),
                 ellipsoid_invalid_args()),
                (faceted_ellipsoid_integrator(),
                 faceted_ellipsoid_valid_args(),
                 faceted_ellipsoid_invalid_args()),
                (faceted_ellipsoid_union_integrator(),
                 faceted_ellipsoid_union_valid_args(),
                 faceted_ellipsoid_union_invalid_args()),
                (polyhedron_integrator(),
                 polyhedron_valid_args(),
                 polyhedron_invalid_args()),
                (simple_polygon_integrator(),
                 simple_polygon_valid_args(),
                 simple_polygon_invalid_args()),
                (sphere_integrator(),
                 sphere_valid_args(),
                 sphere_invalid_args()),
                (sphere_union_integrator(),
                 sphere_union_valid_args(),
                 sphere_union_invalid_args()),
                (convex_spheropolyhedron_union_integrator(),
                 convex_spheropolyhedron_union_valid_args(),
                 convex_spheropolyhedron_union_invalid_args()),
                (sphinx_integrator(),
                 sphinx_valid_args(),
                 sphinx_invalid_args())]
    return get_args
