import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc

def test_ellipsoid():

    args_1 = {'a': 1, 'b': 2, 'c': 3, 'ignore_statistics':1}
    test_ellipsoid1 = hpmc.ell_params(args_1)
    test_dict1 = test_ellipsoid1.asDict()
    assert test_dict1 == args_1
    
    args_2 = {'a': 4, 'b': 1, 'c': 30, 'ignore_statistics':1}
    test_ellipsoid2 = hpmc.ell_params(args_2)
    test_dict2 = test_ellipsoid2.asDict()
    assert test_dict2 == args_2
    
    args_3 = {'a': 10, 'b': 5, 'c': 6, 'ignore_statistics':0}
    test_ellipsoid3 = hpmc.ell_params(args_3)
    test_dict3 = test_ellipsoid3.asDict()
    assert test_dict3 == args_3


