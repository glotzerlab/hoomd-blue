import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc

def test_faceted_ellipsoid():

    args_1 = {"normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1)], "offsets": [1, 3, 2, 6, 3, 1], "a": 3, "b": 4, "c": 1, "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (1, 1, 0)], "origin": (0, 0, 0), "ignore_statistics":1}
    test_faceted_ellipsoid1 = hpmc.faceted_ellipsoid_params(args_1)
    test_dict1 = test_faceted_ellipsoid1.asDict()
    assert test_dict1 == args_1
    
    args_2 = {"normals": [(0, 0, 0), (2, 1, 1), (1, 3, 3), (5, 1, 1), (1, 3, 0), (1, 2, 2)], "offsets": [1, 3, 3, 2, 3, 1], "a": 2, "b": 1, "c": 3, "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2), (0, 0, 1)], "origin": (0, 0, 1), "ignore_statistics":0}
    test_faceted_ellipsoid2 = hpmc.faceted_ellipsoid_params(args_2)
    test_dict2 = test_faceted_ellipsoid2.asDict()
    assert test_dict2 == args_2
    
    args_3 = {"normals": [(0, 0, 2), (0, 1, 1), (1, 3, 5), (0, 1, 6)], "offsets": [6, 2, 2, 5], "a": 1, "b": 6, "c": 6, "vertices": [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)], "origin": (0, 1, 0), "ignore_statistics":1}
    test_faceted_ellipsoid3 = hpmc.faceted_ellipsoid_params(args_3)
    test_dict3 = test_faceted_ellipsoid3.asDict()
    assert test_dict3 == args_3
    
    args_4 = {"normals": [(0, 0, 2), (2, 2, 0), (3, 1, 1), (4, 1, 1), (1, 2, 0), (3, 3, 1), (1, 2, 1), (3, 3, 2)], "offsets": [5, 3, 3, 4, 3, 4, 2, 2], "a": 2, "b": 2, "c": 4, "vertices": [(0, 1, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1)], "origin": (1, 0, 0), "ignore_statistics":0}
    test_faceted_ellipsoid4 = hpmc.faceted_ellipsoid_params(args_4)
    test_dict4 = test_faceted_ellipsoid4.asDict()
    assert test_dict4 == args_4

    args_5 = {"normals": [(0, 0, 1), (0, 4, 0), (2, 0, 1), (0, 3, 1), (4, 1, 0), (2, 2, 1), (1, 3, 1), (1, 9, 0), (2, 2, 2)], "offsets": [5, 4, 2, 2, 7, 3, 1, 4, 1], "a": 6, "b": 1, "c": 1, "vertices": [(0, 10, 3), (3, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 0), (5, 0, 1), (0, 10, 1), (9, 5, 1), (0, 0, 1)], "origin": (0, 0, 0), "ignore_statistics":1}
    test_faceted_ellipsoid5 = hpmc.faceted_ellipsoid_params(args_5)
    test_dict5 = test_faceted_ellipsoid5.asDict()
    assert test_dict5 == args_5

