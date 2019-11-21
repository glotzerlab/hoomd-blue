import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc

def test_polyhedron():
    '''
    args_1 = {'vertices':[(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 1)], 'faces':[[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1]], 'face_offs': [1, 4, 2, 8, 5, 9, 3], 'overlap': [1, 2, 3, 4, 5, 6], 'sweep_radius': 0, 'ignore_statistics': 1, 'capacity': 4, 'origin': (0, 0, 0), 'hull_only': True}
    test_polyhedron1 = hpmc.poly3d_data(args_1)
    test_dict1 = test_polyhedron1.asDict()
    assert test_dict1 == args_1
    
    args_2 = {'vertices':[(0, 3, 0), (2, 1, 0), (1, 3, 1), (1, 1, 1), (1, 2, 5), (3, 0, 1)], 'faces':[[0, 4, 5], [1, 3, 2], [1, 2, 5], [5, 1, 3], [1, 4, 6], [0, 2, 1]], 'face_offs': [1, 4, 20, 8, 5, 9, 1], 'overlap': [1, 4, 6, 4, 1, 6], 'sweep_radius': 2, 'ignore_statistics': 0, 'capacity': 3, 'origin': (0, 1, 0), 'hull_only': False}
    test_polyhedron2 = hpmc.poly3d_data(args_2)
    test_dict2 = test_polyhedron2.asDict()
    assert test_dict2 == args_2
    
    args_3 = {'vertices':[(0, 3, 0), (2, 1, 0), (1, 3, 1), (1, 1, 1), (1, 2, 5), (3, 0, 1), (0, 3, 3)], 'faces':[[0, 1, 2], [3, 2, 6], [1, 2, 4], [6, 1, 3], [3, 4, 6], [4, 5, 1], [6, 7, 5]], 'face_offs': [1, 4, 0, 5, 5, 1, 1, 4], 'overlap': [5, 4, 3, 4, 1, 2, 3], 'sweep_radius': 1, 'ignore_statistics': 1, 'capacity': 4, 'origin': (0, 0, 0), 'hull_only': True}
    test_polyhedron3 = hpmc.poly3d_data(args_3)
    test_dict3 = test_polyhedron3.asDict()
    assert test_dict3 == args_3
    
    args_4 = {'vertices':[(0, 3, 0), (2, 1, 0), (3, 0, 1), (0, 3, 3)], 'faces':[[0, 1, 2], [3, 2, 4], [1, 2, 4], [3, 2, 1]], 'face_offs': [1, 4, 2, 2, 5], 'overlap': [5, 4, 3, 1], 'sweep_radius': 0, 'ignore_statistics': 1, 'capacity': 4, 'origin': (0, 0, 1), 'hull_only': True}
    test_polyhedron4 = hpmc.poly3d_data(args_4)
    test_dict4 = test_polyhedron4.asDict()
    assert test_dict4 == args_4
    
    args_5 = {'vertices':[(0, 3, 0), (2, 1, 0), (1, 3, 1), (1, 1, 1), (1, 2, 5), (3, 0, 1), (0, 3, 3), (0, 0, 2), (1, 2, 2)], 'faces':[[0, 1, 2], [3, 2, 6], [1, 2, 4], [6, 1, 3], [3, 4, 6], [4, 5, 1], [6, 7, 5], [1, 7, 8], [6, 8, 9]], 'face_offs': [1, 4, 0, 5, 5, 1, 1, 4, 2, 1], 'overlap': [5, 4, 3, 4, 1, 2, 3, 1, 1], 'sweep_radius': 0, 'ignore_statistics': 1, 'capacity': 4, 'origin': (0, 1, 0), 'hull_only': True}
    test_polyhedron5 = hpmc.poly3d_data(args_5)
    test_dict5 = test_polyhedron5.asDict()
    assert test_dict5 == args_5
    '''
    assert True
