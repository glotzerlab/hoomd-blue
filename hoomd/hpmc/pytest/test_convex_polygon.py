import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc

def test_convex_polygon():

    args_1 = {'vertices':[(0, 0), (1, 1), (1, 0), (0, 1), (1, 1), (0, 0)], 'ignore_statistics':1}
    test_convex_polygon1 = hpmc.poly2d_verts(args_1)
    test_dict1 = test_convex_polygon1.asDict()
    assert test_dict1 == args_1
    
    args_2 = {'vertices':[(0, 0), (0, 1), (1, 3), (5, 1)], 'ignore_statistics':0}
    test_convex_polygon2 = hpmc.poly2d_verts(args_2)
    test_dict2 = test_convex_polygon2.asDict()
    assert test_dict2 == args_2
    
    args_3 = {'vertices':[(0, 0), (1, 1), (1, 0), (0, 1), (1, 1), (0, 0), (2, 1), (1, 3)], 'ignore_statistics':1}
    test_convex_polygon3 = hpmc.poly2d_verts(args_3)
    test_dict3 = test_convex_polygon3.asDict()
    assert test_dict3 == args_3
    
    args_4 = {'vertices':[(0, 0), (1, 1), (1, 0), (0, 1), (1, 1), (0, 0), (2, 1), (1, 3), (9, 8), (1, 1)], 'ignore_statistics':0}
    test_convex_polygon4 = hpmc.poly2d_verts(args_4)
    test_dict4 = test_convex_polygon4.asDict()
    assert test_dict4 == args_4
