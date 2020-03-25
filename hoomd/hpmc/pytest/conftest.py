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
                      'ignore_statistics': 0,
                      'sweep_radius': 0},
                     {'vertices': [(-0.5, -0.5),
                                   (0.5, -0.5),
                                   (0.5, 0.5),
                                   (-0.5, 0.5)],
                      'ignore_statistics': 0,
                      'sweep_radius': 0.0},
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
        args_list = [{'vertices': "invalid",
                      'ignore_statistics': 0,
                      'sweep_radius': 1.0},
                     {'vertices': 1,
                      'ignore_statistics': 0,
                      'sweep_radius': 3.0},
                     {'vertices': [(0, 0), (1, 1), (1, 0), (0, 1),
                                   (1, 1), (0, 0), (2, 1), (1, 3)],
                      'ignore_statistics': 1,
                      'sweep_radius': "invalid"},
                     {'vertices': [(0, 0), (1, 1), (1, 0), (0, 1),
                                   (1, 1), (0, 0), (2, 1), (1, 3)],
                      'ignore_statistics': "invalid",
                      'sweep_radius': "invalid"}]
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
                                   (0, 0, 0.5)],
                      'ignore_statistics': 0,
                      'sweep_radius': 0},
                     {'vertices': [(0, 5, 0), (1, 1, 1), (1, 0, 1),
                                   (0, 1, 1), (1, 1, 0), (0, 0, 1)],
                      'ignore_statistics': 1,
                      'sweep_radius': 2.0},
                     {'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                      'ignore_statistics': 0,
                      'sweep_radius': 1.0},
                     {'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2),
                                   (2, 1, 1)],
                      'ignore_statistics': 1,
                      'sweep_radius': 0}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_polyhedron_invalid_args():
    def get_args():
        args_list = [{'vertices': "invalid",
                      'ignore_statistics': 1,
                      'sweep_radius': 2.0},
                     {'vertices': 1,
                      'ignore_statistics': 0,
                      'sweep_radius': 1.0},
                     {'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2),
                                   (2, 1, 1)],
                      'ignore_statistics': 1,
                      'sweep_radius': "invalid"},
                     {'vertices': [(0, 0, 0), (1, 1, 1), (1, 0, 2),
                                   (2, 1, 1)],
                      'ignore_statistics': "invalid",
                      'sweep_radius': "invalid"}]
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
        args_list = [{"a": 0.75, "b": 1, "c": 0.5, 'ignore_statistics': 0},
                     {'a': 1, 'b': 2, 'c': 3, 'ignore_statistics': 1},
                     {'a': 4, 'b': 1, 'c': 30, 'ignore_statistics': 1},
                     {'a': 10, 'b': 5, 'c': 6, 'ignore_statistics': 0}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def ellipsoid_invalid_args():
    def get_args():
        args_list = [{'a': 'invalid', 'b': 'invalid', 'c': 'invalid'},
                     {'a': 1, 'b': 3, 'c': 'invalid'},
                     {'a': [1, 2, 3], 'b': [3, 7, 7], 'c': [2, 5, 9]},
                     {'a': 'invalid', 'b': 'invalid', 'c': [1, 2, 3]}]
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
                      "offsets": [0],
                      "ignore_statistics": 0},
                     {"normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0),
                                  (0, 1, 1), (1, 1, 0), (1, 0, 1)],
                      "offsets": [1, 3, 2, 6, 3, 1],
                      "a": 3,
                      "b": 4,
                      "c": 1,
                      "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0),
                                   (1, 0, 0), (1, 1, 1), (1, 1, 0)],
                      "origin": (0, 0, 0),
                      "ignore_statistics": 1},
                     {"normals": [(0, 0, 0), (2, 1, 1), (1, 3, 3),
                                  (5, 1, 1), (1, 3, 0), (1, 2, 2)],
                      "offsets": [1, 3, 3, 2, 3, 1],
                      "a": 2,
                      "b": 1,
                      "c": 3,
                      "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                      "origin": (0, 0, 1),
                      "ignore_statistics": 0},
                     {"normals": [(0, 0, 2), (0, 1, 1),
                                  (1, 3, 5), (0, 1, 6)],
                      "offsets": [6, 2, 2, 5],
                      "a": 1,
                      "b": 6,
                      "c": 6,
                      "vertices": [(0, 0, 0), (1, 1, 1),
                                   (1, 0, 2), (2, 1, 1)],
                      "origin": (0, 1, 0),
                      "ignore_statistics": 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def faceted_ellipsoid_invalid_args():
    def get_args():
        args_list = [{"normals": "invalid",
                      "offsets": [1, 3, 2, 6, 3, 1],
                      "a": 3,
                      "b": 4,
                      "c": 1,
                      "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0),
                                   (1, 0, 0), (1, 1, 1), (1, 1, 0)],
                      "origin": (0, 0, 0),
                      "ignore_statistics": 1},
                     {"normals": 1,
                      "offsets": [1, 3, 3, 2, 3, 1],
                      "a": 2,
                      "b": 1,
                      "c": 3,
                      "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                      "origin": (0, 0, 1),
                      "ignore_statistics": 0},
                     {"normals": [(0, 0, 2), (0, 1, 1),
                                  (1, 3, 5), (0, 1, 6)],
                      "offsets": [6, 2, 2, 5],
                      "a": 1,
                      "b": 6,
                      "c": 6,
                      "vertices": "invalid",
                      "origin": (0, 1, 0),
                      "ignore_statistics": 1},
                     {"normals": [(0, 0, 2), (2, 2, 0), (3, 1, 1),
                                  (4, 1, 1), (1, 2, 0), (3, 3, 1),
                                  (1, 2, 1), (3, 3, 2)],
                      "offsets": [5, 3, 3, 4, 3, 4, 2, 2],
                      "a": 2,
                      "b": 2,
                      "c": 4,
                      "vertices": 3,
                      "origin": (1, 0, 0),
                      "ignore_statistics": 0},
                     {"normals": [(0, 0, 1), (0, 4, 0), (2, 0, 1),
                                  (0, 3, 1), (4, 1, 0), (2, 2, 1),
                                  (1, 3, 1), (1, 9, 0), (2, 2, 2)],
                      "offsets": [5, 4, 2, 2, 7, 3, 1, 4, 1],
                      "a": "invalid",
                      "b": 1,
                      "c": 1,
                      "vertices": [(0, 10, 3), (3, 2, 1), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 0), (5, 0, 1),
                                   (0, 10, 1), (9, 5, 1), (0, 0, 1)],
                      "origin": (0, 0, 0),
                      "ignore_statistics": 1},
                     {"normals": [(0, 0, 1), (0, 4, 0), (2, 0, 1),
                                  (0, 3, 1), (4, 1, 0), (2, 2, 1),
                                  (1, 3, 1), (1, 9, 0), (2, 2, 2)],
                      "offsets": [5, 4, 2, 2, 7, 3, 1, 4, 1],
                      "a": [4, 3, 2],
                      "b": [4, 3, 2],
                      "c": [4, 3, 2],
                      "vertices": [(0, 10, 3), (3, 2, 1), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 0), (5, 0, 1),
                                   (0, 10, 1), (9, 5, 1), (0, 0, 1)],
                      "origin": (0, 0, 0),
                      "ignore_statistics": 1},
                     {"normals": [(0, 0, 1), (0, 4, 0), (2, 0, 1),
                                  (0, 3, 1), (4, 1, 0), (2, 2, 1),
                                  (1, 3, 1), (1, 9, 0), (2, 2, 2)],
                      "offsets": "invalid",
                      "a": 3,
                      "b": 1,
                      "c": 1,
                      "vertices": [(0, 10, 3), (3, 2, 1), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 0), (5, 0, 1),
                                   (0, 10, 1), (9, 5, 1), (0, 0, 1)],
                      "origin": (0, 0, 0),
                      "ignore_statistics": 1}]
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
                      'overlap': [1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [faceted_ell_args_list[2],
                                 faceted_ell_args_list[1]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 0],
                      'capacity': 3,
                      'ignore_statistics': 0},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)],
                      'overlap': [0, 1],
                      'capacity': 5,
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
        args_list = [{'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[1]],
                      'positions': "invalid",
                      'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [faceted_ell_args_list[2],
                                 faceted_ell_args_list[1]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': "invalid",
                      'overlap': [1, 0],
                      'capacity': 3,
                      'ignore_statistics': 0},
                     {'shapes': [faceted_ell_args_list[1],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 1, 0), (0, 0, 1)],
                      'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)],
                      'overlap': "invalid",
                      'capacity': 2,
                      'ignore_statistics': 1},
                     {'shapes': [faceted_ell_args_list[1],
                                 faceted_ell_args_list[0]],
                      'positions': 1,
                      'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)],
                      'overlap': [1, 0],
                      'capacity': 1,
                      'ignore_statistics': 0},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'orientations': 2,
                      'overlap': [0, 1],
                      'capacity': 5,
                      'ignore_statistics': 1},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)],
                      'overlap': [0, 1],
                      'capacity': "invalid",
                      'ignore_statistics': 1},
                     {'shapes': [faceted_ell_args_list[0],
                                 faceted_ell_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)],
                      'overlap': "invalid",
                      'capacity': 5,
                      'ignore_statistics': "invalid"}]
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
                      "overlap": [True, True, True, True, True, True]},
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
                      'sweep_radius': 2,
                      'ignore_statistics': 0,
                      'capacity': 3,
                      'origin': (0, 1, 0),
                      'hull_only': False},
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
                      'overlap': [True, True, True, True, True, True, True],
                      'sweep_radius': 1,
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
                      'sweep_radius': 0,
                      'ignore_statistics': 0,
                      'capacity': 4,
                      'origin': (0, 0, 1),
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
                                (6, 8, 2)],
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': 4,
                      'origin': (0, 1, 0),
                      'hull_only': True}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def polyhedron_invalid_args():
    def get_args():
        args_list = [{'vertices': "invalid",
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
                      'faces': "invalid",
                      'overlap': [0, 0, 0, 0, 0, 0],
                      'sweep_radius': 2,
                      'ignore_statistics': 0,
                      'capacity': 3,
                      'origin': (0, 1, 0),
                      'hull_only': False},
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
                      'overlap': "invalid",
                      'sweep_radius': 1,
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
                      'sweep_radius': "invalid",
                      'ignore_statistics': 0,
                      'capacity': 4,
                      'origin': (0, 0, 1),
                      'hull_only': True},
                     {'vertices': 1,
                      'faces': [(0, 1, 2),
                                (3, 2, 6),
                                (1, 2, 4),
                                (6, 1, 3),
                                (3, 4, 6),
                                (4, 5, 1),
                                (6, 7, 5),
                                (1, 7, 8),
                                (6, 8, 2)],
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': 4,
                      'origin': (0, 1, 0),
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
                      'faces': 1,
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': 4,
                      'origin': (0, 1, 0),
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
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': 1,
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': 4,
                      'origin': (0, 1, 0),
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
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': "invalid",
                      'capacity': "invalid",
                      'origin': (0, 1, 0),
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
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': "invalid",
                      'origin': (0, 1, 0),
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
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': "invalid",
                      'origin': "invalid",
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
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': 4,
                      'origin': 1,
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
                      'faces': [(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
                      'overlap': [0, 1, 1, 0, 0, 0, 1, 1, 0],
                      'sweep_radius': 0,
                      'ignore_statistics': 1,
                      'capacity': "invalid",
                      'origin': (0, 0, 0),
                      'hull_only': "invalid"}]
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
        args_list = [{"vertices": "invalid"},
                     {"vertices": 1},
                     # {"vertices": [(-1, 1), (1, -1), (1, 1)],
                     #  "ignore_statistics": "invalid"},
                     {"vertices": [(-1, 1), (1, -1), (1, 1)],
                      "sweep_radius": "invalid"}]
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
        args_list = [{"diameter": 1, 'orientable': 0, 'ignore_statistics': 0},
                     {'diameter': 1, 'orientable': 0, 'ignore_statistics': 1},
                     {'diameter': 9, 'orientable': 1, 'ignore_statistics': 1},
                     {'diameter': 4, 'orientable': 0, 'ignore_statistics': 0}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphere_invalid_args():
    def get_args():
        args_list = [{"diameter": "invalid"},
                     {"diameter": [1, 2, 3, 4]}]
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
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)],
                      'overlap': [1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[0], sphere_args_list[2]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 2],
                      'capacity': 3,
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[2], sphere_args_list[1]],
                      'positions': [(1, 1, 0), (1, 0, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 0],
                      'capacity': 2,
                      'ignore_statistics': 0},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1],
                                 sphere_args_list[2]],
                      'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1)],
                      'orientations': [(1, 1, 1, 0),
                                       (1, 1, 0, 0),
                                       (1, 0, 0, 1)],
                      'overlap': [1, 1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphere_union_invalid_args(sphere_valid_args):
    def get_args():
        sphere_args_list = sphere_valid_args()
        args_list = [{'shapes': [sphere_args_list[0], sphere_args_list[1]],
                      'positions': "invalid",
                      'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)],
                      'overlap': [1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[0], sphere_args_list[2]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': "invalid",
                      'overlap': [1, 2],
                      'capacity': 3,
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[2], sphere_args_list[1]],
                      'positions': [(1, 1, 0), (1, 0, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                      'overlap': "invalid",
                      'capacity': 2,
                      'ignore_statistics': 0},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1],
                                 sphere_args_list[2]],
                      'positions': 1,
                      'orientations': [(1, 1, 1, 0),
                                       (1, 1, 0, 0),
                                       (1, 0, 0, 1)],
                      'overlap': [1, 1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1],
                                 sphere_args_list[2]],
                      'positions': [(1, 1, 0), (1, 0, 1)],
                      'orientations': 1,
                      'overlap': [1, 1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1],
                                 sphere_args_list[2]],
                      'positions': [(1, 1, 0), (1, 0, 1)],
                      'orientations': [(1, 1, 1, 0),
                                       (1, 1, 0, 0),
                                       (1, 0, 0, 1)],
                      'overlap': 1,
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1],
                                 sphere_args_list[2]],
                      'positions': [(1, 1, 0), (1, 0, 1)],
                      'orientations': [(1, 1, 1, 0),
                                       (1, 1, 0, 0),
                                       (1, 0, 0, 1)],
                      'overlap': [1, 1, 1],
                      'capacity': "invalid",
                      'ignore_statistics': 1},
                     {'shapes': [sphere_args_list[0], sphere_args_list[1],
                                 sphere_args_list[2]],
                      'positions': [(1, 1, 0), (1, 0, 1)],
                      'orientations': [(1, 1, 1, 0),
                                       (1, 1, 0, 0),
                                       (1, 0, 0, 1)],
                      'overlap': [1, 1, 1],
                      'capacity': "invalid",
                      'ignore_statistics': "invalid"}]
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
                      'positions': [(0, 0, 0), (0, 0, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [polyhedron_vertices_args_list[2],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 0],
                      'capacity': 3,
                      'ignore_statistics': 0},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)],
                      'overlap': [0, 1],
                      'capacity': 5,
                      'ignore_statistics': 1},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1],
                                 polyhedron_vertices_args_list[2]],
                      'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 1, 1, 1),
                                       (1, 0, 0, 1)],
                      'overlap': [True, True, False],
                      'capacity': 4,
                      'ignore_statistics': 0}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def convex_spheropolyhedron_union_invalid_args(convex_polyhedron_valid_args):
    def get_args():
        polyhedron_vertices_args_list = convex_polyhedron_valid_args()
        args_list = [{'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': "invalid",
                      'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                      'overlap': [1, 1],
                      'capacity': 4,
                      'ignore_statistics': 1},
                     {'shapes': [polyhedron_vertices_args_list[2],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(1, 0, 0), (0, 0, 1)],
                      'orientations': "invalid",
                      'overlap': [1, 2],
                      'capacity': 3,
                      'ignore_statistics': 0},
                     {'shapes': [polyhedron_vertices_args_list[1],
                                 polyhedron_vertices_args_list[0]],
                      'positions': [(1, 1, 0), (0, 0, 1)],
                      'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)],
                      'overlap': "invalid",
                      'capacity': 2,
                      'ignore_statistics': 1},
                     {'shapes': [polyhedron_vertices_args_list[1],
                                 polyhedron_vertices_args_list[2]],
                      'positions': 1,
                      'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)],
                      'overlap': [1, 0],
                      'capacity': 1,
                      'ignore_statistics': 0},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[2]],
                      'positions': [(1, 0, 1), (0, 0, 0)],
                      'orientations': 2,
                      'overlap': [0, 1],
                      'capacity': 5,
                      'ignore_statistics': 1},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(1, 0, 0), (0, 1, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)],
                      'overlap': 2,
                      'capacity': 6,
                      'ignore_statistics': 0},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(1, 0, 0), (0, 1, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)],
                      'overlap': [0, 1],
                      'capacity': "invalid",
                      'ignore_statistics': 0},
                     {'shapes': [polyhedron_vertices_args_list[0],
                                 polyhedron_vertices_args_list[1]],
                      'positions': [(1, 0, 0), (0, 1, 1)],
                      'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)],
                      'overlap': [0, 1],
                      'capacity': "invalid",
                      'ignore_statistics': "invalid"}]
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
                      'centers': [(0, 0, 0), (0.5, 0, 0)],
                      'ignore_statistics': 0},
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
                                  (3, 5, 3)],
                      'ignore_statistics': 1}]
        return args_list
    return get_args


@pytest.fixture(scope="function")
def sphinx_invalid_args():
    def get_args():
        args_list = [{'diameters': "invalid",
                      'centers': [(0, 0, 0),
                                  (1, 1, 1),
                                  (1, 0, 1),
                                  (0, 1, 1),
                                  (1, 1, 0),
                                  (0, 0, 1)],
                      'ignore_statistics': 1},
                     {'diameters': [5, 2, 4, 5, 1, 2],
                      'centers': "invalid",
                      'ignore_statistics': 0},
                     {'diameters': 2,
                      'centers': [(0, 0, 0),
                                  (1, 1, 1),
                                  (1, 0, 1),
                                  (0, 1, 1),
                                  (1, 1, 0),
                                  (0, 0, 1),
                                  (2, 2, 1),
                                  (3, 5, 3)],
                      'ignore_statistics': 1},
                     {'diameters': [1, 4, 2, 8, 5],
                      'centers': 4,
                      'ignore_statistics': 0}]
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
