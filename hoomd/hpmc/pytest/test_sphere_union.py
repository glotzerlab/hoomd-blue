import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc

def test_dict_conversions():
    sph_args_1 = {'diameter': 1, 'orientable': 0, 'ignore_statistics': 1}
    sph_args_2 = {'diameter': 9, 'orientable': 1, 'ignore_statistics': 1}
    sph_args_3 = {'diameter': 4, 'orientable': 0, 'ignore_statistics': 0}

    sph_union_args1 = {'shapes': [sph_args_1, sph_args_2],
                       'positions': [(0, 0, 0), (0, 0, 1)],
                       'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                       'overlap': [1, 1],
                       'capacity': 4,
                       'ignore_statistics': 1}
    test_sph_union1 = _hpmc.SphereUnionParams(sph_union_args1)
    test_sph_dict1 = test_sph_union1.asDict()
    assert test_sph_dict1 == sph_union_args1

    sph_union_args2 = {'shapes': [sph_args_1, sph_args_3],
                       'positions': [(1, 0, 0), (0, 0, 1)],
                       'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                       'overlap': [1, 2],
                       'capacity': 3,
                       'ignore_statistics': 1}
    test_sph_union2 = _hpmc.SphereUnionParams(sph_union_args2)
    test_sph_dict2 = test_sph_union2.asDict()
    assert test_sph_dict2 == sph_union_args2

    sph_union_args3 = {'shapes': [sph_args_3, sph_args_2],
                       'positions': [(1, 1, 0), (1, 0, 1)],
                       'orientations': [(1, 0, 0, 1), (1, 0, 1, 0)],
                       'overlap': [1, 0],
                       'capacity': 2,
                       'ignore_statistics': 0}
    test_sph_union3 = _hpmc.SphereUnionParams(sph_union_args3)
    test_sph_dict3 = test_sph_union3.asDict()
    assert test_sph_dict3 == sph_union_args3

    sph_union_args4 = {'shapes': [sph_args_1, sph_args_2, sph_args_3],
                       'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1)],
                       'orientations': [(1, 1, 0, 0),
                                        (1, 0, 0, 0),
                                        (1, 0, 1, 0)],
                       'overlap': [1, 1, 1],
                       'capacity': 4,
                       'ignore_statistics': 1}
    test_sph_union4 = _hpmc.SphereUnionParams(sph_union_args4)
    test_sph_dict4 = test_sph_union4.asDict()
    assert test_sph_dict4 == sph_union_args4
