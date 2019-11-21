import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc

def test_union():

    sph_args_1 = {'diameter':1, 'orientable': 0, 'ignore_statistics':1}
    sph_args_2 = {'diameter':9, 'orientable': 1, 'ignore_statistics':1}
    sph_args_3 = {'diameter':4, 'orientable': 0, 'ignore_statistics':0}
    
    test_sphere1 = hpmc.sph_params(sph_args_1)
    test_sphere2 = hpmc.sph_params(sph_args_2)
    test_sphere3 = hpmc.sph_params(sph_args_3)
    
    sph_union_args1 = {'members': [test_sphere1, test_sphere2], 'positions': [(0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1], 'capacity': 4, 'ignore_statistics': 1}
    sph_test_args1 = {'members': [sph_args_1, sph_args_2], 'positions': [(0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1], 'capacity': 4, 'ignore_statistics': 1}
    test_sph_union1 = hpmc.msph_params(sph_union_args1)
    test_sph_dict1 = test_sph_union1.asDict()
    assert test_sph_dict1 == sph_test_args1
    
    sph_union_args2 = {'members': [test_sphere1, test_sphere3], 'positions': [(1, 0, 0), (0, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 2], 'capacity': 3, 'ignore_statistics': 1}
    sph_test_args2 = {'members': [sph_args_1, sph_args_3], 'positions': [(1, 0, 0), (0, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 2], 'capacity': 3, 'ignore_statistics': 1}
    test_sph_union2 = hpmc.msph_params(sph_union_args2)
    test_sph_dict2 = test_sph_union2.asDict()
    assert test_sph_dict2 == sph_test_args2
    
    sph_union_args3 = {'members': [test_sphere3, test_sphere2], 'positions': [(1, 1, 0), (1, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 0, 1, 0)], 'overlap': [1, 0], 'capacity': 2, 'ignore_statistics': 0}
    sph_test_args3 = {'members': [sph_args_3, sph_args_2], 'positions': [(1, 1, 0), (1, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 0, 1, 0)], 'overlap': [1, 0], 'capacity': 2, 'ignore_statistics': 0}
    test_sph_union3 = hpmc.msph_params(sph_union_args3)
    test_sph_dict3 = test_sph_union3.asDict()
    assert test_sph_dict3 == sph_test_args3
    
    sph_union_args4 = {'members': [test_sphere1, test_sphere2, test_sphere3], 'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    sph_test_args4 = {'members': [sph_args_1, sph_args_2, sph_args_3], 'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    test_sph_union4 = hpmc.msph_params(sph_union_args4)
    test_sph_dict4 = test_sph_union4.asDict()
    assert test_sph_dict4 == sph_test_args4
    
    
    polyhedron_args_1 = {'vertices':[(0, 5, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1)], 'ignore_statistics':1}
    polyhedron_args_2 = {'vertices':[(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2), (0, 0, 1)], 'ignore_statistics':0}
    polyhedron_args_3 = {'vertices':[(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)], 'ignore_statistics':1}
    polyhedron_args_4 = {'vertices':[(0, 1, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1)], 'ignore_statistics':0}
    polyhedron_args_5 = {'vertices':[(0, 10, 3), (3, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 0), (5, 0, 1), (0, 10, 1), (9, 5, 1), (0, 0, 1)], 'ignore_statistics':1}
    
    test_polyhedron1 = hpmc.poly3d_verts(polyhedron_args_1)
    test_polyhedron2 = hpmc.poly3d_verts(polyhedron_args_2)
    test_polyhedron3 = hpmc.poly3d_verts(polyhedron_args_3)
    test_polyhedron4 = hpmc.poly3d_verts(polyhedron_args_4)
    test_polyhedron5 = hpmc.poly3d_verts(polyhedron_args_5)
    
    polyhedron_union_args1 = {'members': [test_polyhedron1, test_polyhedron2], 'positions': [(0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args2 = {'members': [test_polyhedron3, test_polyhedron2], 'positions': [(1, 0, 0), (0, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 2], 'capacity': 3, 'ignore_statistics': 0}
    polyhedron_union_args3 = {'members': [test_polyhedron4, test_polyhedron2], 'positions': [(1, 1, 0), (0, 0, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [1, 3], 'capacity': 2, 'ignore_statistics': 1}
    polyhedron_union_args4 = {'members': [test_polyhedron5, test_polyhedron2], 'positions': [(1, 1, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 0], 'capacity': 1, 'ignore_statistics': 0}
    polyhedron_union_args5 = {'members': [test_polyhedron1, test_polyhedron3], 'positions': [(1, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1], 'capacity': 5, 'ignore_statistics': 1}
    polyhedron_union_args6 = {'members': [test_polyhedron1, test_polyhedron4], 'positions': [(1, 0, 0), (0, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [2, 1], 'capacity': 6, 'ignore_statistics': 0}
    polyhedron_union_args7 = {'members': [test_polyhedron1, test_polyhedron5], 'positions': [(0, 1, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [3, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args8 = {'members': [test_polyhedron3, test_polyhedron4], 'positions': [(0, 0, 0), (1, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)], 'overlap': [0, 0], 'capacity': 4, 'ignore_statistics': 0}
    polyhedron_union_args9 = {'members': [test_polyhedron3, test_polyhedron5], 'positions': [(0, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 1, 0), (1, 0, 1, 0)], 'overlap': [2, 2], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args10 = {'members': [test_polyhedron4, test_polyhedron5], 'positions': [(0, 1, 0), (1, 0, 1)], 'orientations': [(1, 1, 0, 1), (1, 0, 0, 0)], 'overlap': [3, 3], 'capacity': 4, 'ignore_statistics': 0}
    
    polyhedron_union_args11 = {'members': [test_polyhedron1, test_polyhedron2, test_polyhedron3], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args12 = {'members': [test_polyhedron1, test_polyhedron3, test_polyhedron4], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1], 'capacity': 3, 'ignore_statistics': 0}
    polyhedron_union_args13 = {'members': [test_polyhedron1, test_polyhedron4, test_polyhedron5], 'positions': [(1, 1, 0), (0, 0, 1), (0, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [1, 3, 0], 'capacity': 2, 'ignore_statistics': 1}
    polyhedron_union_args14 = {'members': [test_polyhedron2, test_polyhedron3, test_polyhedron4], 'positions': [(1, 1, 1), (0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 0, 2], 'capacity': 4, 'ignore_statistics': 0}
    polyhedron_union_args15 = {'members': [test_polyhedron2, test_polyhedron4, test_polyhedron5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args16 = {'members': [test_polyhedron3, test_polyhedron4, test_polyhedron5], 'positions': [(0, 1, 0), (1, 0, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [2, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    
    polyhedron_union_args17 = {'members': [test_polyhedron1, test_polyhedron2, test_polyhedron3, test_polyhedron4], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args18 = {'members': [test_polyhedron1, test_polyhedron2, test_polyhedron3, test_polyhedron5], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 2, 0], 'capacity': 3, 'ignore_statistics': 0}
    polyhedron_union_args19 = {'members': [test_polyhedron1, test_polyhedron2, test_polyhedron4, test_polyhedron5], 'positions': [(1, 1, 0), (1, 0, 1), (0, 0, 0), (1, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1, 1], 'capacity': 2, 'ignore_statistics': 1}
    polyhedron_union_args20 = {'members': [test_polyhedron1, test_polyhedron3, test_polyhedron4, test_polyhedron5], 'positions': [(1, 1, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    polyhedron_union_args21 = {'members': [test_polyhedron2, test_polyhedron3, test_polyhedron4, test_polyhedron5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 2, 2, 2], 'capacity': 4, 'ignore_statistics': 1}
    
    polyhedron_union_args22 = {'members': [test_polyhedron1, test_polyhedron2, test_polyhedron3, test_polyhedron4, test_polyhedron5], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0), (2, 2, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0)], 'overlap': [1, 1, 0, 2, 1], 'capacity': 4, 'ignore_statistics': 1}
    
    
    polyhedron_test_args1 = {'members': [polyhedron_args_1, polyhedron_args_2], 'positions': [(0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_test_args2 = {'members': [polyhedron_args_3, polyhedron_args_2], 'positions': [(1, 0, 0), (0, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 2], 'capacity': 3, 'ignore_statistics': 0}
    polyhedron_test_args3 = {'members': [polyhedron_args_4, polyhedron_args_2], 'positions': [(1, 1, 0), (0, 0, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [1, 3], 'capacity': 2, 'ignore_statistics': 1}
    polyhedron_test_args4 = {'members': [polyhedron_args_5, polyhedron_args_2], 'positions': [(1, 1, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 0], 'capacity': 1, 'ignore_statistics': 0}
    polyhedron_test_args5 = {'members': [polyhedron_args_1, polyhedron_args_3], 'positions': [(1, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1], 'capacity': 5, 'ignore_statistics': 1}
    polyhedron_test_args6 = {'members': [polyhedron_args_1, polyhedron_args_4], 'positions': [(1, 0, 0), (0, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [2, 1], 'capacity': 6, 'ignore_statistics': 0}
    polyhedron_test_args7 = {'members': [polyhedron_args_1, polyhedron_args_5], 'positions': [(0, 1, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [3, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_test_args8 = {'members': [polyhedron_args_3, polyhedron_args_4], 'positions': [(0, 0, 0), (1, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)], 'overlap': [0, 0], 'capacity': 4, 'ignore_statistics': 0}
    polyhedron_test_args9 = {'members': [polyhedron_args_3, polyhedron_args_5], 'positions': [(0, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 1, 0), (1, 0, 1, 0)], 'overlap': [2, 2], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_test_args10 = {'members': [polyhedron_args_4, polyhedron_args_5], 'positions': [(0, 1, 0), (1, 0, 1)], 'orientations': [(1, 1, 0, 1), (1, 0, 0, 0)], 'overlap': [3, 3], 'capacity': 4, 'ignore_statistics': 0}
    
    polyhedron_test_args11 = {'members': [polyhedron_args_1, polyhedron_args_2, polyhedron_args_3], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_test_args12 = {'members': [polyhedron_args_1, polyhedron_args_3, polyhedron_args_4], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1], 'capacity': 3, 'ignore_statistics': 0}
    polyhedron_test_args13 = {'members': [polyhedron_args_1, polyhedron_args_4, polyhedron_args_5], 'positions': [(1, 1, 0), (0, 0, 1), (0, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [1, 3, 0], 'capacity': 2, 'ignore_statistics': 1}
    polyhedron_test_args14 = {'members': [polyhedron_args_2, polyhedron_args_3, polyhedron_args_4], 'positions': [(1, 1, 1), (0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 0, 2], 'capacity': 4, 'ignore_statistics': 0}
    polyhedron_test_args15 = {'members': [polyhedron_args_2, polyhedron_args_4, polyhedron_args_5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_test_args16 = {'members': [polyhedron_args_3, polyhedron_args_4, polyhedron_args_5], 'positions': [(0, 1, 0), (1, 0, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [2, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    
    polyhedron_test_args17 = {'members': [polyhedron_args_1, polyhedron_args_2, polyhedron_args_3, polyhedron_args_4], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    polyhedron_test_args18 = {'members': [polyhedron_args_1, polyhedron_args_2, polyhedron_args_3, polyhedron_args_5], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 2, 0], 'capacity': 3, 'ignore_statistics': 0}
    polyhedron_test_args19 = {'members': [polyhedron_args_1, polyhedron_args_2, polyhedron_args_4, polyhedron_args_5], 'positions': [(1, 1, 0), (1, 0, 1), (0, 0, 0), (1, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1, 1], 'capacity': 2, 'ignore_statistics': 1}
    polyhedron_test_args20 = {'members': [polyhedron_args_1, polyhedron_args_3, polyhedron_args_4, polyhedron_args_5], 'positions': [(1, 1, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    polyhedron_test_args21 = {'members': [polyhedron_args_2, polyhedron_args_3, polyhedron_args_4, polyhedron_args_5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 2, 2, 2], 'capacity': 4, 'ignore_statistics': 1}
    
    polyhedron_test_args22 = {'members': [polyhedron_args_1, polyhedron_args_2, polyhedron_args_3, polyhedron_args_4, polyhedron_args_5], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0), (2, 2, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0)], 'overlap': [1, 1, 0, 2, 1], 'capacity': 4, 'ignore_statistics': 1}
    
    test_polyhedron_union1 = hpmc.mpoly3d_params(polyhedron_union_args1)
    test_polyhedron_dict1 = test_polyhedron_union1.asDict()
    test_polyhedron_union2 = hpmc.mpoly3d_params(polyhedron_union_args2)
    test_polyhedron_dict2 = test_polyhedron_union2.asDict()
    test_polyhedron_union3 = hpmc.mpoly3d_params(polyhedron_union_args3)
    test_polyhedron_dict3 = test_polyhedron_union3.asDict()
    test_polyhedron_union4 = hpmc.mpoly3d_params(polyhedron_union_args4)
    test_polyhedron_dict4 = test_polyhedron_union4.asDict()
    test_polyhedron_union5 = hpmc.mpoly3d_params(polyhedron_union_args5)
    test_polyhedron_dict5 = test_polyhedron_union5.asDict()
    test_polyhedron_union6 = hpmc.mpoly3d_params(polyhedron_union_args6)
    test_polyhedron_dict6 = test_polyhedron_union6.asDict()
    test_polyhedron_union7 = hpmc.mpoly3d_params(polyhedron_union_args7)
    test_polyhedron_dict7 = test_polyhedron_union7.asDict()
    test_polyhedron_union8 = hpmc.mpoly3d_params(polyhedron_union_args8)
    test_polyhedron_dict8 = test_polyhedron_union8.asDict()
    test_polyhedron_union9 = hpmc.mpoly3d_params(polyhedron_union_args9)
    test_polyhedron_dict9 = test_polyhedron_union9.asDict()
    test_polyhedron_union10 = hpmc.mpoly3d_params(polyhedron_union_args10)
    test_polyhedron_dict10 = test_polyhedron_union10.asDict()
    test_polyhedron_union11 = hpmc.mpoly3d_params(polyhedron_union_args11)
    test_polyhedron_dict11 = test_polyhedron_union11.asDict()
    test_polyhedron_union12 = hpmc.mpoly3d_params(polyhedron_union_args12)
    test_polyhedron_dict12 = test_polyhedron_union12.asDict()
    test_polyhedron_union13 = hpmc.mpoly3d_params(polyhedron_union_args13)
    test_polyhedron_dict13 = test_polyhedron_union13.asDict()
    test_polyhedron_union14 = hpmc.mpoly3d_params(polyhedron_union_args14)
    test_polyhedron_dict14 = test_polyhedron_union14.asDict()
    test_polyhedron_union15 = hpmc.mpoly3d_params(polyhedron_union_args15)
    test_polyhedron_dict15 = test_polyhedron_union15.asDict()
    test_polyhedron_union16 = hpmc.mpoly3d_params(polyhedron_union_args16)
    test_polyhedron_dict16 = test_polyhedron_union16.asDict()
    test_polyhedron_union17 = hpmc.mpoly3d_params(polyhedron_union_args17)
    test_polyhedron_dict17 = test_polyhedron_union17.asDict()
    test_polyhedron_union18 = hpmc.mpoly3d_params(polyhedron_union_args18)
    test_polyhedron_dict18 = test_polyhedron_union18.asDict()
    test_polyhedron_union19 = hpmc.mpoly3d_params(polyhedron_union_args19)
    test_polyhedron_dict19 = test_polyhedron_union19.asDict()
    test_polyhedron_union20 = hpmc.mpoly3d_params(polyhedron_union_args20)
    test_polyhedron_dict20 = test_polyhedron_union20.asDict()
    test_polyhedron_union21 = hpmc.mpoly3d_params(polyhedron_union_args21)
    test_polyhedron_dict21 = test_polyhedron_union21.asDict()
    test_polyhedron_union22 = hpmc.mpoly3d_params(polyhedron_union_args22)
    test_polyhedron_dict22 = test_polyhedron_union22.asDict()
    
    assert test_polyhedron_dict1 == polyhedron_test_args1
    assert test_polyhedron_dict2 == polyhedron_test_args2
    assert test_polyhedron_dict3 == polyhedron_test_args3
    assert test_polyhedron_dict4 == polyhedron_test_args4
    assert test_polyhedron_dict5 == polyhedron_test_args5
    assert test_polyhedron_dict6 == polyhedron_test_args6
    assert test_polyhedron_dict7 == polyhedron_test_args7
    assert test_polyhedron_dict8 == polyhedron_test_args8
    assert test_polyhedron_dict9 == polyhedron_test_args9
    assert test_polyhedron_dict10 == polyhedron_test_args10
    assert test_polyhedron_dict11 == polyhedron_test_args11
    assert test_polyhedron_dict12 == polyhedron_test_args12
    assert test_polyhedron_dict13 == polyhedron_test_args13
    assert test_polyhedron_dict14 == polyhedron_test_args14
    assert test_polyhedron_dict15 == polyhedron_test_args15
    assert test_polyhedron_dict16 == polyhedron_test_args16
    assert test_polyhedron_dict17 == polyhedron_test_args17
    assert test_polyhedron_dict18 == polyhedron_test_args18
    assert test_polyhedron_dict19 == polyhedron_test_args19
    assert test_polyhedron_dict20 == polyhedron_test_args20
    assert test_polyhedron_dict21 == polyhedron_test_args21
    assert test_polyhedron_dict22 == polyhedron_test_args22
    
    
    
    
    
    faceted_ell_args_1 = {"normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1)], "offsets": [1, 3, 2, 6, 3, 1], "a": 3, "b": 4, "c": 1, "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (1, 1, 0)], "origin": (0, 0, 0), "ignore_statistics":1}
    faceted_ell_args_2 = {"normals": [(0, 0, 0), (2, 1, 1), (1, 3, 3), (5, 1, 1), (1, 3, 0), (1, 2, 2)], "offsets": [1, 3, 3, 2, 3, 1], "a": 2, "b": 1, "c": 3, "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2), (0, 0, 1)], "origin": (0, 0, 1), "ignore_statistics":0}
    faceted_ell_args_3 = {"normals": [(0, 0, 2), (0, 1, 1), (1, 3, 5), (0, 1, 6)], "offsets": [6, 2, 2, 5], "a": 1, "b": 6, "c": 6, "vertices": [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)], "origin": (0, 1, 0), "ignore_statistics":1}
    faceted_ell_args_4 = {"normals": [(0, 0, 2), (2, 2, 0), (3, 1, 1), (4, 1, 1), (1, 2, 0), (3, 3, 1), (1, 2, 1), (3, 3, 2)], "offsets": [5, 3, 3, 4, 3, 4, 2, 2], "a": 2, "b": 2, "c": 4, "vertices": [(0, 1, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1)], "origin": (1, 0, 0), "ignore_statistics":0}
    faceted_ell_args_5 = {"normals": [(0, 0, 1), (0, 4, 0), (2, 0, 1), (0, 3, 1), (4, 1, 0), (2, 2, 1), (1, 3, 1), (1, 9, 0), (2, 2, 2)], "offsets": [5, 4, 2, 2, 7, 3, 1, 4, 1], "a": 6, "b": 1, "c": 1, "vertices": [(0, 10, 3), (3, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 0), (5, 0, 1), (0, 10, 1), (9, 5, 1), (0, 0, 1)], "origin": (0, 0, 0), "ignore_statistics":1}
    
    test_faceted_ell1 = hpmc.faceted_ellipsoid_params(faceted_ell_args_1)
    test_faceted_ell2 = hpmc.faceted_ellipsoid_params(faceted_ell_args_2)
    test_faceted_ell3 = hpmc.faceted_ellipsoid_params(faceted_ell_args_3)
    test_faceted_ell4 = hpmc.faceted_ellipsoid_params(faceted_ell_args_4)
    test_faceted_ell5 = hpmc.faceted_ellipsoid_params(faceted_ell_args_5)
    
    faceted_ell_union_args1 = {'members': [test_faceted_ell1, test_faceted_ell2], 'positions': [(0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_union_args2 = {'members': [test_faceted_ell3, test_faceted_ell2], 'positions': [(1, 0, 0), (0, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 2], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_union_args3 = {'members': [test_faceted_ell4, test_faceted_ell2], 'positions': [(1, 1, 0), (0, 0, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [1, 3], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_union_args4 = {'members': [test_faceted_ell5, test_faceted_ell2], 'positions': [(1, 1, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 0], 'capacity': 1, 'ignore_statistics': 0}
    faceted_ell_union_args5 = {'members': [test_faceted_ell1, test_faceted_ell3], 'positions': [(1, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1], 'capacity': 5, 'ignore_statistics': 1}
    faceted_ell_union_args6 = {'members': [test_faceted_ell1, test_faceted_ell4], 'positions': [(1, 0, 0), (0, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [2, 1], 'capacity': 6, 'ignore_statistics': 0}
    faceted_ell_union_args7 = {'members': [test_faceted_ell1, test_faceted_ell5], 'positions': [(0, 1, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [3, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_union_args8 = {'members': [test_faceted_ell3, test_faceted_ell4], 'positions': [(0, 0, 0), (1, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)], 'overlap': [0, 0], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_union_args9 = {'members': [test_faceted_ell3, test_faceted_ell5], 'positions': [(0, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 1, 0), (1, 0, 1, 0)], 'overlap': [2, 2], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_union_args10 = {'members': [test_faceted_ell4, test_faceted_ell5], 'positions': [(0, 1, 0), (1, 0, 1)], 'orientations': [(1, 1, 0, 1), (1, 0, 0, 0)], 'overlap': [3, 3], 'capacity': 4, 'ignore_statistics': 0}
    
    faceted_ell_union_args11 = {'members': [test_faceted_ell1, test_faceted_ell2, test_faceted_ell3], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_union_args12 = {'members': [test_faceted_ell1, test_faceted_ell3, test_faceted_ell4], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_union_args13 = {'members': [test_faceted_ell1, test_faceted_ell4, test_faceted_ell5], 'positions': [(1, 1, 0), (0, 0, 1), (0, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [1, 3, 0], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_union_args14 = {'members': [test_faceted_ell2, test_faceted_ell3, test_faceted_ell4], 'positions': [(1, 1, 1), (0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 0, 2], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_union_args15 = {'members': [test_faceted_ell2, test_faceted_ell4, test_faceted_ell5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_union_args16 = {'members': [test_faceted_ell3, test_faceted_ell4, test_faceted_ell5], 'positions': [(0, 1, 0), (1, 0, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [2, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    
    faceted_ell_union_args17 = {'members': [test_faceted_ell1, test_faceted_ell2, test_faceted_ell3, test_faceted_ell4], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_union_args18 = {'members': [test_faceted_ell1, test_faceted_ell2, test_faceted_ell3, test_faceted_ell5], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 2, 0], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_union_args19 = {'members': [test_faceted_ell1, test_faceted_ell2, test_faceted_ell4, test_faceted_ell5], 'positions': [(1, 1, 0), (1, 0, 1), (0, 0, 0), (1, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1, 1], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_union_args20 = {'members': [test_faceted_ell1, test_faceted_ell3, test_faceted_ell4, test_faceted_ell5], 'positions': [(1, 1, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_union_args21 = {'members': [test_faceted_ell2, test_faceted_ell3, test_faceted_ell4, test_faceted_ell5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 2, 2, 2], 'capacity': 4, 'ignore_statistics': 1}
    
    faceted_ell_union_args22 = {'members': [test_faceted_ell1, test_faceted_ell2, test_faceted_ell3, test_faceted_ell4, test_faceted_ell5], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0), (2, 2, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0)], 'overlap': [1, 1, 0, 2, 1], 'capacity': 4, 'ignore_statistics': 1}
    
    
    faceted_ell_test_args1 = {'members': [faceted_ell_args_1, faceted_ell_args_2], 'positions': [(0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args2 = {'members': [faceted_ell_args_3, faceted_ell_args_2], 'positions': [(1, 0, 0), (0, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 2], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_test_args3 = {'members': [faceted_ell_args_4, faceted_ell_args_2], 'positions': [(1, 1, 0), (0, 0, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [1, 3], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_test_args4 = {'members': [faceted_ell_args_5, faceted_ell_args_2], 'positions': [(1, 1, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 0], 'capacity': 1, 'ignore_statistics': 0}
    faceted_ell_test_args5 = {'members': [faceted_ell_args_1, faceted_ell_args_3], 'positions': [(1, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1], 'capacity': 5, 'ignore_statistics': 1}
    faceted_ell_test_args6 = {'members': [faceted_ell_args_1, faceted_ell_args_4], 'positions': [(1, 0, 0), (0, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [2, 1], 'capacity': 6, 'ignore_statistics': 0}
    faceted_ell_test_args7 = {'members': [faceted_ell_args_1, faceted_ell_args_5], 'positions': [(0, 1, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [3, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args8 = {'members': [faceted_ell_args_3, faceted_ell_args_4], 'positions': [(0, 0, 0), (1, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)], 'overlap': [0, 0], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_test_args9 = {'members': [faceted_ell_args_3, faceted_ell_args_5], 'positions': [(0, 0, 1), (0, 0, 0)], 'orientations': [(1, 0, 1, 0), (1, 0, 1, 0)], 'overlap': [2, 2], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args10 = {'members': [faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 1, 0), (1, 0, 1)], 'orientations': [(1, 1, 0, 1), (1, 0, 0, 0)], 'overlap': [3, 3], 'capacity': 4, 'ignore_statistics': 0}
    
    faceted_ell_test_args11 = {'members': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args12 = {'members': [faceted_ell_args_1, faceted_ell_args_3, faceted_ell_args_4], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_test_args13 = {'members': [faceted_ell_args_1, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(1, 1, 0), (0, 0, 1), (0, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [1, 3, 0], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_test_args14 = {'members': [faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4], 'positions': [(1, 1, 1), (0, 0, 0), (0, 0, 1)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 0, 2], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_test_args15 = {'members': [faceted_ell_args_2, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args16 = {'members': [faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 1, 0), (1, 0, 1), (0, 0, 1)], 'orientations': [(1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 0)], 'overlap': [2, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    
    faceted_ell_test_args17 = {'members': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)], 'overlap': [1, 1, 1, 1], 'capacity': 4, 'ignore_statistics': 1}
    faceted_ell_test_args18 = {'members': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_5], 'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)], 'orientations': [(1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 1)], 'overlap': [1, 1, 2, 0], 'capacity': 3, 'ignore_statistics': 0}
    faceted_ell_test_args19 = {'members': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(1, 1, 0), (1, 0, 1), (0, 0, 0), (1, 1, 1)], 'orientations': [(1, 0, 1, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0)], 'overlap': [1, 2, 1, 1], 'capacity': 2, 'ignore_statistics': 1}
    faceted_ell_test_args20 = {'members': [faceted_ell_args_1, faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(1, 1, 1), (0, 0, 1), (0, 0, 0), (1, 1, 0)], 'orientations': [(1, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)], 'overlap': [0, 1, 1, 0], 'capacity': 4, 'ignore_statistics': 0}
    faceted_ell_test_args21 = {'members': [faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1), (1, 0, 0)], 'orientations': [(1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0, 1), (1, 0, 0, 0)], 'overlap': [1, 2, 2, 2], 'capacity': 4, 'ignore_statistics': 1}
    
    faceted_ell_test_args22 = {'members': [faceted_ell_args_1, faceted_ell_args_2, faceted_ell_args_3, faceted_ell_args_4, faceted_ell_args_5], 'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1), (1, 1, 0), (2, 2, 0)], 'orientations': [(1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0)], 'overlap': [1, 1, 0, 2, 1], 'capacity': 4, 'ignore_statistics': 1}
    
    test_faceted_ell_union1 = hpmc.mfellipsoid_params(faceted_ell_union_args1)
    test_faceted_ell_dict1 = test_faceted_ell_union1.asDict()
    test_faceted_ell_union2 = hpmc.mfellipsoid_params(faceted_ell_union_args2)
    test_faceted_ell_dict2 = test_faceted_ell_union2.asDict()
    test_faceted_ell_union3 = hpmc.mfellipsoid_params(faceted_ell_union_args3)
    test_faceted_ell_dict3 = test_faceted_ell_union3.asDict()
    test_faceted_ell_union4 = hpmc.mfellipsoid_params(faceted_ell_union_args4)
    test_faceted_ell_dict4 = test_faceted_ell_union4.asDict()
    test_faceted_ell_union5 = hpmc.mfellipsoid_params(faceted_ell_union_args5)
    test_faceted_ell_dict5 = test_faceted_ell_union5.asDict()
    test_faceted_ell_union6 = hpmc.mfellipsoid_params(faceted_ell_union_args6)
    test_faceted_ell_dict6 = test_faceted_ell_union6.asDict()
    test_faceted_ell_union7 = hpmc.mfellipsoid_params(faceted_ell_union_args7)
    test_faceted_ell_dict7 = test_faceted_ell_union7.asDict()
    test_faceted_ell_union8 = hpmc.mfellipsoid_params(faceted_ell_union_args8)
    test_faceted_ell_dict8 = test_faceted_ell_union8.asDict()
    test_faceted_ell_union9 = hpmc.mfellipsoid_params(faceted_ell_union_args9)
    test_faceted_ell_dict9 = test_faceted_ell_union9.asDict()
    test_faceted_ell_union10 = hpmc.mfellipsoid_params(faceted_ell_union_args10)
    test_faceted_ell_dict10 = test_faceted_ell_union10.asDict()
    test_faceted_ell_union11 = hpmc.mfellipsoid_params(faceted_ell_union_args11)
    test_faceted_ell_dict11 = test_faceted_ell_union11.asDict()
    test_faceted_ell_union12 = hpmc.mfellipsoid_params(faceted_ell_union_args12)
    test_faceted_ell_dict12 = test_faceted_ell_union12.asDict()
    test_faceted_ell_union13 = hpmc.mfellipsoid_params(faceted_ell_union_args13)
    test_faceted_ell_dict13 = test_faceted_ell_union13.asDict()
    test_faceted_ell_union14 = hpmc.mfellipsoid_params(faceted_ell_union_args14)
    test_faceted_ell_dict14 = test_faceted_ell_union14.asDict()
    test_faceted_ell_union15 = hpmc.mfellipsoid_params(faceted_ell_union_args15)
    test_faceted_ell_dict15 = test_faceted_ell_union15.asDict()
    test_faceted_ell_union16 = hpmc.mfellipsoid_params(faceted_ell_union_args16)
    test_faceted_ell_dict16 = test_faceted_ell_union16.asDict()
    test_faceted_ell_union17 = hpmc.mfellipsoid_params(faceted_ell_union_args17)
    test_faceted_ell_dict17 = test_faceted_ell_union17.asDict()
    test_faceted_ell_union18 = hpmc.mfellipsoid_params(faceted_ell_union_args18)
    test_faceted_ell_dict18 = test_faceted_ell_union18.asDict()
    test_faceted_ell_union19 = hpmc.mfellipsoid_params(faceted_ell_union_args19)
    test_faceted_ell_dict19 = test_faceted_ell_union19.asDict()
    test_faceted_ell_union20 = hpmc.mfellipsoid_params(faceted_ell_union_args20)
    test_faceted_ell_dict20 = test_faceted_ell_union20.asDict()
    test_faceted_ell_union21 = hpmc.mfellipsoid_params(faceted_ell_union_args21)
    test_faceted_ell_dict21 = test_faceted_ell_union21.asDict()
    test_faceted_ell_union22 = hpmc.mfellipsoid_params(faceted_ell_union_args22)
    test_faceted_ell_dict22 = test_faceted_ell_union22.asDict()
    
    assert test_faceted_ell_dict1 == faceted_ell_test_args1
    assert test_faceted_ell_dict2 == faceted_ell_test_args2
    assert test_faceted_ell_dict3 == faceted_ell_test_args3
    assert test_faceted_ell_dict4 == faceted_ell_test_args4
    assert test_faceted_ell_dict5 == faceted_ell_test_args5
    assert test_faceted_ell_dict6 == faceted_ell_test_args6
    assert test_faceted_ell_dict7 == faceted_ell_test_args7
    assert test_faceted_ell_dict8 == faceted_ell_test_args8
    assert test_faceted_ell_dict9 == faceted_ell_test_args9
    assert test_faceted_ell_dict10 == faceted_ell_test_args10
    assert test_faceted_ell_dict11 == faceted_ell_test_args11
    assert test_faceted_ell_dict12 == faceted_ell_test_args12
    assert test_faceted_ell_dict13 == faceted_ell_test_args13
    assert test_faceted_ell_dict14 == faceted_ell_test_args14
    assert test_faceted_ell_dict15 == faceted_ell_test_args15
    assert test_faceted_ell_dict16 == faceted_ell_test_args16
    assert test_faceted_ell_dict17 == faceted_ell_test_args17
    assert test_faceted_ell_dict18 == faceted_ell_test_args18
    assert test_faceted_ell_dict19 == faceted_ell_test_args19
    assert test_faceted_ell_dict20 == faceted_ell_test_args20
    assert test_faceted_ell_dict21 == faceted_ell_test_args21
    assert test_faceted_ell_dict22 == faceted_ell_test_args22
    
        
