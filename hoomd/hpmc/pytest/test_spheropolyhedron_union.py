import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc

def test_dict_conversions():

    polyhedron_args_1 = {'vertices': [(0, 5, 0), (1, 1, 1), (1, 0, 1),
                                      (0, 1, 1), (1, 1, 0), (0, 0, 1)],
                         'ignore_statistics': 1,
                         'sweep_radius': 0}
    polyhedron_args_2 = {'vertices': [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                      (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                         'ignore_statistics': 0,
                         'sweep_radius': 1}
    polyhedron_args_3 = {'vertices': [(0, 0, 0), (1, 1, 1),
                                      (1, 0, 2), (2, 1, 1)],
                         'ignore_statistics': 1,
                         'sweep_radius': 3.125}
    polyhedron_args_4 = {'vertices' :[(0, 1, 0), (1, 1, 1), (1, 0, 1),
                                      (0, 1, 1), (1, 1, 0), (0, 0, 1),
                                      (0, 0, 1), (0, 0, 1)],
                         'ignore_statistics': 0,
                         'sweep_radius': 5.5}
    polyhedron_args_5 = {'vertices': [(0, 10, 3), (3, 2, 1), (1, 2, 1),
                                      (0, 1, 1), (1, 1, 0), (5, 0, 1),
                                      (0, 10, 1), (9, 5, 1), (0, 0, 1)],
                         'ignore_statistics': 1,
                         'sweep_radius': 6.25}

    polyhedron_union_args1 = {'shapes': [polyhedron_args_1, polyhedron_args_2],
                              'positions': [(0, 0, 0), (0, 0, 1)],
                              'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                              'overlap': [1, 1],
                              'capacity': 4,
                              'ignore_statistics': 1}
    polyhedron_union_args2 = {'shapes': [polyhedron_args_3, polyhedron_args_2],
                              'positions': [(1, 0, 0), (0, 0, 1)],
                              'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                              'overlap': [1, 2],
                              'capacity': 3,
                              'ignore_statistics': 0}
    polyhedron_union_args3 = {'shapes': [polyhedron_args_4, polyhedron_args_2],
                              'positions': [(1, 1, 0), (0, 0, 1)],
                              'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)],
                              'overlap': [1, 3],
                              'capacity': 2,
                              'ignore_statistics': 1}
    polyhedron_union_args4 = {'shapes': [polyhedron_args_5, polyhedron_args_2],
                              'positions': [(1, 1, 1), (0, 0, 0)],
                              'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)],
                              'overlap': [1, 0],
                              'capacity': 1,
                              'ignore_statistics': 0}
    polyhedron_union_args5 = {'shapes': [polyhedron_args_1, polyhedron_args_3],
                              'positions': [(1, 0, 1), (0, 0, 0)],
                              'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)],
                              'overlap': [0, 1],
                              'capacity': 5,
                              'ignore_statistics': 1}
    polyhedron_union_args6 = {'shapes': [polyhedron_args_1, polyhedron_args_4],
                              'positions': [(1, 0, 0), (0, 1, 1)],
                              'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)],
                              'overlap': [2, 1],
                              'capacity': 6,
                              'ignore_statistics': 0}
    polyhedron_union_args7 = {'shapes': [polyhedron_args_1, polyhedron_args_5],
                              'positions': [(0, 1, 1), (0, 0, 1)],
                              'orientations': [(1, 0, 0, 0), (1, 0, 0, 1)],
                              'overlap': [3, 1],
                              'capacity': 4,
                              'ignore_statistics': 1}
    polyhedron_union_args8 = {'shapes': [polyhedron_args_3, polyhedron_args_4],
                              'positions': [(0, 0, 0), (1, 0, 1)],
                              'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)],
                              'overlap': [0, 0],
                              'capacity': 4,
                              'ignore_statistics': 0}
    polyhedron_union_args9 = {'shapes': [polyhedron_args_3, polyhedron_args_5],
                              'positions': [(0, 0, 1), (0, 0, 0)],
                              'orientations': [(1, 0, 1, 0), (1, 0, 1, 0)],
                              'overlap': [2, 2],
                              'capacity': 4,
                              'ignore_statistics': 1}
    polyhedron_union_args10 = {'shapes': [polyhedron_args_4, polyhedron_args_5],
                               'positions': [(0, 1, 0), (1, 0, 1)],
                               'orientations': [(1, 1, 0, 1), (1, 0, 0, 0)],
                               'overlap': [3, 3],
                               'capacity': 4,
                               'ignore_statistics': 0}
    polyhedron_union_args11 = {'shapes': [polyhedron_args_1, polyhedron_args_2,
                                          polyhedron_args_3],
                               'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)],
                               'orientations': [(1, 0, 0, 0), (1, 0, 0, 0),
                                                (1, 0, 0, 1)],
                               'overlap': [1, 1, 1],
                               'capacity': 4,
                               'ignore_statistics': 1}
    polyhedron_union_args12 = {'shapes': [polyhedron_args_1, polyhedron_args_3,
                                          polyhedron_args_4],
                               'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1)],
                               'orientations': [(1, 1, 0, 0), (1, 0, 0, 0),
                                                (1, 0, 1, 0)],
                               'overlap': [1, 2, 1],
                               'capacity': 3,
                               'ignore_statistics': 0}
    polyhedron_union_args13 = {'shapes': [polyhedron_args_1, polyhedron_args_4,
                                          polyhedron_args_5],
                               'positions': [(1, 1, 0), (0, 0, 1), (0, 1, 1)],
                               'orientations': [(1, 0, 1, 0), (1, 0, 0, 0),
                                                (1, 1, 0, 0)],
                               'overlap': [1, 3, 0],
                               'capacity': 2, 'ignore_statistics': 1}
    polyhedron_union_args14 = {'shapes': [polyhedron_args_2, polyhedron_args_3,
                                          polyhedron_args_4],
                               'positions': [(1, 1, 1), (0, 0, 0), (0, 0, 1)],
                               'orientations': [(1, 0, 0, 1), (1, 0, 0, 0),
                                                (1, 0, 0, 0)],
                               'overlap': [1, 0, 2],
                               'capacity': 4, 'ignore_statistics': 0}
    polyhedron_union_args15 = {'shapes': [polyhedron_args_2, polyhedron_args_4,
                                          polyhedron_args_5],
                               'positions': [(0, 0, 0), (0, 1, 1), (1, 0, 1)],
                               'orientations': [(1, 0, 0, 0), (1, 1, 0, 0),
                                                (1, 1, 0, 0)],
                               'overlap': [0, 1, 1],
                               'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args16 = {'shapes': [polyhedron_args_3, polyhedron_args_4,
                                          polyhedron_args_5],
                               'positions': [(0, 1, 0), (1, 0, 1), (0, 0, 1)],
                               'orientations': [(1, 0, 0, 0), (1, 0, 1, 0),
                                                (1, 0, 0, 0)],
                               'overlap': [2, 1, 0],
                               'capacity': 4, 'ignore_statistics': 0}
    polyhedron_union_args17 = {'shapes': [polyhedron_args_1, polyhedron_args_2,
                                          polyhedron_args_3, polyhedron_args_4],
                               'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1),
                                             (1, 1, 0)],
                               'orientations': [(1, 0, 0, 0), (1, 0, 0, 0),
                                                (1, 0, 0, 0), (1, 0, 0, 0)],
                               'overlap': [1, 1, 1, 1],
                               'capacity': 4, 'ignore_statistics': 1}
    polyhedron_union_args18 = {'shapes': [polyhedron_args_1, polyhedron_args_2,
                                          polyhedron_args_3, polyhedron_args_5],
                               'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1),
                                             (0, 1, 1)],
                               'orientations': [(1, 1, 0, 0), (1, 0, 0, 0),
                                                (1, 0, 0, 0), (1, 0, 0, 1)],
                               'overlap': [1, 1, 2, 0],
                               'capacity': 3, 'ignore_statistics': 0}
    polyhedron_union_args19 = {'shapes': [polyhedron_args_1, polyhedron_args_2,
                                          polyhedron_args_4, polyhedron_args_5],
                               'positions': [(1, 1, 0), (1, 0, 1), (0, 0, 0),
                                             (1, 1, 1)],
                               'orientations': [(1, 0, 1, 0), (1, 0, 0, 0),
                                                (1, 0, 0, 0), (1, 0, 1, 0)],
                               'overlap': [1, 2, 1, 1],
                               'capacity': 2, 'ignore_statistics': 1}
    polyhedron_union_args20 = {'shapes': [polyhedron_args_1, polyhedron_args_3,
                                          polyhedron_args_4, polyhedron_args_5],
                               'positions': [(1, 1, 1), (0, 0, 1), (0, 0, 0),
                                             (1, 1, 0)],
                               'orientations': [(1, 0, 0, 1), (1, 0, 0, 0),
                                                (1, 0, 0, 0), (1, 1, 0, 0)],
                               'overlap': [0, 1, 1, 0],
                               'capacity': 4, 'ignore_statistics': 0}
    polyhedron_union_args21 = {'shapes': [polyhedron_args_2, polyhedron_args_3,
                                          polyhedron_args_4, polyhedron_args_5],
                               'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1),
                                             (1, 0, 0)],
                               'orientations': [(1, 0, 0, 0), (1, 1, 0, 0),
                                                (1, 0, 0, 1), (1, 0, 0, 0)],
                               'overlap': [1, 2, 2, 2],
                               'capacity': 4, 'ignore_statistics': 1}

    polyhedron_union_args22 = {'shapes': [polyhedron_args_1, polyhedron_args_2,
                                          polyhedron_args_3, polyhedron_args_4,
                                          polyhedron_args_5],
                               'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1),
                                             (1, 1, 0), (2, 2, 0)],
                               'orientations': [(1, 0, 0, 0), (1, 0, 0, 0),
                                                (1, 1, 0, 0), (0, 0, 1, 1),
                                                (1, 0, 0, 0)],
                               'overlap': [1, 1, 0, 0, 1],
                               'capacity': 4, 'ignore_statistics': 1}

    polyhedron_args__union1 = _hpmc.mpoly3d_params(polyhedron_union_args1)
    polyhedron_args__dict1 = polyhedron_args__union1.asDict()
    polyhedron_args__union2 = _hpmc.mpoly3d_params(polyhedron_union_args2)
    polyhedron_args__dict2 = polyhedron_args__union2.asDict()
    polyhedron_args__union3 = _hpmc.mpoly3d_params(polyhedron_union_args3)
    polyhedron_args__dict3 = polyhedron_args__union3.asDict()
    polyhedron_args__union4 = _hpmc.mpoly3d_params(polyhedron_union_args4)
    polyhedron_args__dict4 = polyhedron_args__union4.asDict()
    polyhedron_args__union5 = _hpmc.mpoly3d_params(polyhedron_union_args5)
    polyhedron_args__dict5 = polyhedron_args__union5.asDict()
    polyhedron_args__union6 = _hpmc.mpoly3d_params(polyhedron_union_args6)
    polyhedron_args__dict6 = polyhedron_args__union6.asDict()
    polyhedron_args__union7 = _hpmc.mpoly3d_params(polyhedron_union_args7)
    polyhedron_args__dict7 = polyhedron_args__union7.asDict()
    polyhedron_args__union8 = _hpmc.mpoly3d_params(polyhedron_union_args8)
    polyhedron_args__dict8 = polyhedron_args__union8.asDict()
    polyhedron_args__union9 = _hpmc.mpoly3d_params(polyhedron_union_args9)
    polyhedron_args__dict9 = polyhedron_args__union9.asDict()
    polyhedron_args__union10 = _hpmc.mpoly3d_params(polyhedron_union_args10)
    polyhedron_args__dict10 = polyhedron_args__union10.asDict()
    polyhedron_args__union11 = _hpmc.mpoly3d_params(polyhedron_union_args11)
    polyhedron_args__dict11 = polyhedron_args__union11.asDict()
    polyhedron_args__union12 = _hpmc.mpoly3d_params(polyhedron_union_args12)
    polyhedron_args__dict12 = polyhedron_args__union12.asDict()
    polyhedron_args__union13 = _hpmc.mpoly3d_params(polyhedron_union_args13)
    polyhedron_args__dict13 = polyhedron_args__union13.asDict()
    polyhedron_args__union14 = _hpmc.mpoly3d_params(polyhedron_union_args14)
    polyhedron_args__dict14 = polyhedron_args__union14.asDict()
    polyhedron_args__union15 = _hpmc.mpoly3d_params(polyhedron_union_args15)
    polyhedron_args__dict15 = polyhedron_args__union15.asDict()
    polyhedron_args__union16 = _hpmc.mpoly3d_params(polyhedron_union_args16)
    polyhedron_args__dict16 = polyhedron_args__union16.asDict()
    polyhedron_args__union17 = _hpmc.mpoly3d_params(polyhedron_union_args17)
    polyhedron_args__dict17 = polyhedron_args__union17.asDict()
    polyhedron_args__union18 = _hpmc.mpoly3d_params(polyhedron_union_args18)
    polyhedron_args__dict18 = polyhedron_args__union18.asDict()
    polyhedron_args__union19 = _hpmc.mpoly3d_params(polyhedron_union_args19)
    polyhedron_args__dict19 = polyhedron_args__union19.asDict()
    polyhedron_args__union20 = _hpmc.mpoly3d_params(polyhedron_union_args20)
    polyhedron_args__dict20 = polyhedron_args__union20.asDict()
    polyhedron_args__union21 = _hpmc.mpoly3d_params(polyhedron_union_args21)
    polyhedron_args__dict21 = polyhedron_args__union21.asDict()
    polyhedron_args__union22 = _hpmc.mpoly3d_params(polyhedron_union_args22)
    polyhedron_args__dict22 = polyhedron_args__union22.asDict()

    assert polyhedron_args__dict1 == polyhedron_union_args1
    assert polyhedron_args__dict2 == polyhedron_union_args2
    assert polyhedron_args__dict3 == polyhedron_union_args3
    assert polyhedron_args__dict4 == polyhedron_union_args4
    assert polyhedron_args__dict5 == polyhedron_union_args5
    assert polyhedron_args__dict6 == polyhedron_union_args6
    assert polyhedron_args__dict7 == polyhedron_union_args7
    assert polyhedron_args__dict8 == polyhedron_union_args8
    assert polyhedron_args__dict9 == polyhedron_union_args9
    assert polyhedron_args__dict10 == polyhedron_union_args10
    assert polyhedron_args__dict11 == polyhedron_union_args11
    assert polyhedron_args__dict12 == polyhedron_union_args12
    assert polyhedron_args__dict13 == polyhedron_union_args13
    assert polyhedron_args__dict14 == polyhedron_union_args14
    assert polyhedron_args__dict15 == polyhedron_union_args15
    assert polyhedron_args__dict16 == polyhedron_union_args16
    assert polyhedron_args__dict17 == polyhedron_union_args17
    assert polyhedron_args__dict18 == polyhedron_union_args18
    assert polyhedron_args__dict19 == polyhedron_union_args19
    assert polyhedron_args__dict20 == polyhedron_union_args20
    assert polyhedron_args__dict21 == polyhedron_union_args21
    assert polyhedron_args__dict22 == polyhedron_union_args22
