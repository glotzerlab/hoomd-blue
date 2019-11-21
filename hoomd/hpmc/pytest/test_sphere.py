import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc

def test_sphere():

    args_1 = {'diameter':1, 'orientable': 0, 'ignore_statistics':1}
    test_sphere1 = hpmc.sph_params(args_1)
    test_dict1 = test_sphere1.asDict()
    assert test_dict1 == args_1
    
    args_2 = {'diameter':9, 'orientable': 1, 'ignore_statistics':1}
    test_sphere2 = hpmc.sph_params(args_2)
    test_dict2 = test_sphere2.asDict()
    assert test_dict2 == args_2
    
    args_3 = {'diameter':4, 'orientable': 0, 'ignore_statistics':0}
    test_sphere3 = hpmc.sph_params(args_3)
    test_dict3 = test_sphere3.asDict()
    assert test_dict3 == args_3

