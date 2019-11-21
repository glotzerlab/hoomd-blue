import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc

def test_sphinx():

    args_1 = {'diameters':[1, 4, 2, 8, 5, 9], 'centers':[(0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1)], 'ignore_statistics':1}
    test_sphinx1 = hpmc.sphinx3d_params(args_1)
    test_dict1 = test_sphinx1.asDict()
    assert test_dict1 == args_1
    
    args_2 = {'diameters':[5, 2, 4, 5, 1, 2], 'centers':[(0, 2, 0), (1, 4, 1), (3, 0, 1), (3, 1, 1), (1, 4, 0), (2, 2, 1)], 'ignore_statistics':0}
    test_sphinx2 = hpmc.sphinx3d_params(args_2)
    test_dict2 = test_sphinx2.asDict()
    assert test_dict2 == args_2
    
    args_3 = {'diameters':[1, 2, 2, 3, 4, 9, 3, 2], 'centers':[(0, 0, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1), (2, 2, 1), (3, 5, 3)], 'ignore_statistics':1}
    test_sphinx3 = hpmc.sphinx3d_params(args_3)
    test_dict3 = test_sphinx3.asDict()
    assert test_dict3 == args_3
    
    args_4 = {'diameters':[1, 4, 2, 8, 5], 'centers':[(0, 2, 0), (1, 4, 1), (3, 0, 1), (3, 1, 1), (1, 4, 0)], 'ignore_statistics':0}
    test_sphinx4 = hpmc.sphinx3d_params(args_4)
    test_dict4 = test_sphinx4.asDict()
    assert test_dict4 == args_4
