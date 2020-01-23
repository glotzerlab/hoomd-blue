import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
import pytest
import copy


faceted_ell_args_1 = {"normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0),
                                  (0, 1, 1), (1, 1, 0), (1, 0, 1)],
                      "offsets": [1, 3, 2, 6, 3, 1],
                      "a": 3, "b": 4, "c": 1,
                      "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0),
                                   (1, 0, 0), (1, 1, 1), (1, 1, 0)],
                      "origin": (0, 0, 0), "ignore_statistics": 1}

faceted_ell_args_2 = {"normals": [(0, 0, 0), (2, 1, 1), (1, 3, 3),
                                  (5, 1, 1), (1, 3, 0), (1, 2, 2)],
                      "offsets": [1, 3, 3, 2, 3, 1],
                      "a": 2, "b": 1, "c": 3,
                      "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1),
                                   (0, 1, 1), (1, 1, 2), (0, 0, 1)],
                      "origin": (0, 0, 1), "ignore_statistics": 0}

faceted_ell_args_3 = {"normals": [(0, 0, 2), (0, 1, 1), (1, 3, 5), (0, 1, 6)],
                      "offsets": [6, 2, 2, 5],
                      "a": 1,
                      "b": 6,
                      "c": 6,
                      "vertices": [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)],
                      "origin": (0, 1, 0),
                      "ignore_statistics": 1}

faceted_ell_args_4 = {"normals": [(0, 0, 2),
                                  (2, 2, 0),
                                  (3, 1, 1),
                                  (4, 1, 1),
                                  (1, 2, 0),
                                  (3, 3, 1),
                                  (1, 2, 1),
                                  (3, 3, 2)],
                      "offsets": [5, 3, 3, 4, 3, 4, 2, 2],
                      "a": 2,
                      "b": 2,
                      "c": 4,
                      "vertices": [(0, 1, 0),
                                   (1, 1, 1),
                                   (1, 0, 1),
                                   (0, 1, 1),
                                   (1, 1, 0),
                                   (0, 0, 1),
                                   (0, 0, 1),
                                   (0, 0, 1)],
                      "origin": (1, 0, 0),
                      "ignore_statistics": 0}

faceted_ell_args_5 = {"normals": [(0, 0, 1),
                                  (0, 4, 0),
                                  (2, 0, 1),
                                  (0, 3, 1),
                                  (4, 1, 0),
                                  (2, 2, 1),
                                  (1, 3, 1),
                                  (1, 9, 0),
                                  (2, 2, 2)],
                      "offsets": [5, 4, 2, 2, 7, 3, 1, 4, 1],
                      "a": 6,
                      "b": 1,
                      "c": 1,
                      "vertices": [(0, 10, 3),
                                   (3, 2, 1),
                                   (1, 2, 1),
                                   (0, 1, 1),
                                   (1, 1, 0),
                                   (5, 0, 1),
                                   (0, 10, 1),
                                   (9, 5, 1),
                                   (0, 0, 1)],
                      "origin": (0, 0, 0),
                      "ignore_statistics": 1}

faceted_ell_union_args1 = {'shapes': [faceted_ell_args_1, faceted_ell_args_2],
                           'positions': [(0, 0, 0), (0, 0, 1)],
                           'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                           'overlap': [1, 1],
                           'capacity': 4,
                           'ignore_statistics': 1}

faceted_ell_union_args2 = {'shapes': [faceted_ell_args_3, faceted_ell_args_2],
                           'positions': [(1, 0, 0), (0, 0, 1)],
                           'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                           'overlap': [1, 0],
                           'capacity': 3,
                           'ignore_statistics': 0}

faceted_ell_union_args3 = {'shapes': [faceted_ell_args_4, faceted_ell_args_2],
                           'positions': [(1, 1, 0), (0, 0, 1)],
                           'orientations': [(1, 0, 1, 0), (1, 0, 0, 0)],
                           'overlap': [1, 1],
                           'capacity': 2,
                           'ignore_statistics': 1}

faceted_ell_union_args4 = {'shapes': [faceted_ell_args_5, faceted_ell_args_2],
                           'positions': [(1, 1, 1), (0, 0, 0)],
                           'orientations': [(1, 0, 0, 1), (1, 0, 0, 0)],
                           'overlap': [1, 0],
                           'capacity': 1,
                           'ignore_statistics': 0}

faceted_ell_union_args5 = {'shapes': [faceted_ell_args_1, faceted_ell_args_3],
                           'positions': [(1, 0, 1), (0, 0, 0)],
                           'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)],
                           'overlap': [0, 1],
                           'capacity': 5,
                           'ignore_statistics': 1}

faceted_ell_union_args6 = {'shapes': [faceted_ell_args_1, faceted_ell_args_4],
                           'positions': [(1, 0, 0), (0, 1, 1)],
                           'orientations': [(1, 0, 0, 0), (1, 0, 1, 0)],
                           'overlap': [0, 1],
                           'capacity': 6,
                           'ignore_statistics': 0}

faceted_ell_union_args7 = {'shapes': [faceted_ell_args_1, faceted_ell_args_5],
                           'positions': [(0, 1, 1), (0, 0, 1)],
                           'orientations': [(1, 0, 0, 0), (1, 0, 0, 1)],
                           'overlap': [0, 1],
                           'capacity': 4,
                           'ignore_statistics': 1}

faceted_ell_union_args8 = {'shapes': [faceted_ell_args_3, faceted_ell_args_4],
                           'positions': [(0, 0, 0), (1, 0, 1)],
                           'orientations': [(1, 0, 0, 1), (1, 1, 0, 0)],
                           'overlap': [0, 0],
                           'capacity': 4,
                           'ignore_statistics': 0}

faceted_ell_union_args9 = {'shapes': [faceted_ell_args_3, faceted_ell_args_5],
                           'positions': [(0, 0, 1), (0, 0, 0)],
                           'orientations': [(1, 0, 1, 0), (1, 0, 1, 0)],
                           'overlap': [1, 1],
                           'capacity': 4,
                           'ignore_statistics': 1}

faceted_ell_union_args10 = {'shapes': [faceted_ell_args_4, faceted_ell_args_5],
                            'positions': [(0, 1, 0), (1, 0, 1)],
                            'orientations': [(1, 1, 0, 1), (1, 0, 0, 0)],
                            'overlap': [0, 1],
                            'capacity': 4,
                            'ignore_statistics': 0}

faceted_ell_union_args11 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_2,
                                       faceted_ell_args_3],
                            'positions': [(0, 0, 0), (0, 0, 1), (1, 1, 1)],
                            'orientations': [(1, 0, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 1)],
                            'overlap': [1, 1, 1],
                            'capacity': 4,
                            'ignore_statistics': 1}

faceted_ell_union_args12 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_3,
                                       faceted_ell_args_4],
                            'positions': [(1, 0, 0), (0, 0, 1), (1, 0, 1)],
                            'orientations': [(1, 1, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 1, 0)],
                            'overlap': [1, 0, 1],
                            'capacity': 3,
                            'ignore_statistics': 0}

faceted_ell_union_args13 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_4,
                                       faceted_ell_args_5],
                            'positions': [(1, 1, 0), (0, 0, 1), (0, 1, 1)],
                            'orientations': [(1, 0, 1, 0),
                                             (1, 0, 0, 0),
                                             (1, 1, 0, 0)],
                            'overlap': [1, 1, 0],
                            'capacity': 2,
                            'ignore_statistics': 1}

faceted_ell_union_args14 = {'shapes': [faceted_ell_args_2,
                                       faceted_ell_args_3,
                                       faceted_ell_args_4],
                            'positions': [(1, 1, 1), (0, 0, 0), (0, 0, 1)],
                            'orientations': [(1, 0, 0, 1),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 0)],
                            'overlap': [1, 0, 0],
                            'capacity': 4,
                            'ignore_statistics': 0}

faceted_ell_union_args15 = {'shapes': [faceted_ell_args_2,
                                       faceted_ell_args_4,
                                       faceted_ell_args_5],
                            'positions': [(0, 0, 0), (0, 1, 1), (1, 0, 1)],
                            'orientations': [(1, 0, 0, 0),
                                             (1, 1, 0, 0),
                                             (1, 1, 0, 0)],
                            'overlap': [0, 1, 1],
                            'capacity': 4,
                            'ignore_statistics': 1}

faceted_ell_union_args16 = {'shapes': [faceted_ell_args_3,
                                       faceted_ell_args_4,
                                       faceted_ell_args_5],
                            'positions': [(0, 1, 0), (1, 0, 1), (0, 0, 1)],
                            'orientations': [(1, 0, 0, 0),
                                             (1, 0, 1, 0),
                                             (1, 0, 0, 0)],
                            'overlap': [0, 1, 0],
                            'capacity': 4,
                            'ignore_statistics': 0}

faceted_ell_union_args17 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_2,
                                       faceted_ell_args_3,
                                       faceted_ell_args_4],
                            'positions': [(0, 0, 0),
                                          (0, 0, 1),
                                          (1, 1, 1),
                                          (1, 1, 0)],
                            'orientations': [(1, 0, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 0)],
                            'overlap': [1, 1, 1, 1],
                            'capacity': 4,
                            'ignore_statistics': 1}

faceted_ell_union_args18 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_2,
                                       faceted_ell_args_3,
                                       faceted_ell_args_5],
                            'positions': [(1, 0, 0),
                                          (0, 0, 1),
                                          (1, 0, 1),
                                          (0, 1, 1)],
                            'orientations': [(1, 1, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 1)],
                            'overlap': [1, 1, 0, 0],
                            'capacity': 3,
                            'ignore_statistics': 0}

faceted_ell_union_args19 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_2,
                                       faceted_ell_args_4,
                                       faceted_ell_args_5],
                            'positions': [(1, 1, 0),
                                          (1, 0, 1),
                                          (0, 0, 0),
                                          (1, 1, 1)],
                            'orientations': [(1, 0, 1, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 0, 1, 0)],
                            'overlap': [1, 0, 1, 1],
                            'capacity': 2,
                            'ignore_statistics': 1}

faceted_ell_union_args20 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_3,
                                       faceted_ell_args_4,
                                       faceted_ell_args_5],
                            'positions': [(1, 1, 1),
                                          (0, 0, 1),
                                          (0, 0, 0),
                                          (1, 1, 0)],
                            'orientations': [(1, 0, 0, 1),
                                             (1, 0, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 1, 0, 0)],
                            'overlap': [0, 1, 1, 0],
                            'capacity': 4,
                            'ignore_statistics': 0}

faceted_ell_union_args21 = {'shapes': [faceted_ell_args_2,
                                       faceted_ell_args_3,
                                       faceted_ell_args_4,
                                       faceted_ell_args_5],
                            'positions': [(0, 0, 0),
                                          (0, 1, 1),
                                          (1, 1, 1),
                                          (1, 0, 0)],
                            'orientations': [(1, 0, 0, 0),
                                             (1, 1, 0, 0),
                                             (1, 0, 0, 1),
                                             (1, 0, 0, 0)],
                            'overlap': [1, 0, 0, 0],
                            'capacity': 4,
                            'ignore_statistics': 1}

faceted_ell_union_args22 = {'shapes': [faceted_ell_args_1,
                                       faceted_ell_args_2,
                                       faceted_ell_args_3,
                                       faceted_ell_args_4,
                                       faceted_ell_args_5],
                            'positions': [(0, 0, 0),
                                          (0, 0, 1),
                                          (1, 1, 1),
                                          (1, 1, 0),
                                          (2, 2, 0)],
                            'orientations': [(1, 0, 0, 0),
                                             (1, 0, 0, 0),
                                             (1, 1, 0, 0),
                                             (0, 0, 1, 1),
                                             (1, 0, 0, 0)],
                            'overlap': [1, 1, 0, 2, 1],
                            'capacity': 4,
                            'ignore_statistics': 1}


def test_dict_conversions():

    test_facet_ell_union1 = _hpmc.mfellipsoid_params(faceted_ell_union_args1)
    test_facet_ell_dict1 = test_facet_ell_union1.asDict()
    test_facet_ell_union2 = _hpmc.mfellipsoid_params(faceted_ell_union_args2)
    test_facet_ell_dict2 = test_facet_ell_union2.asDict()
    test_facet_ell_union3 = _hpmc.mfellipsoid_params(faceted_ell_union_args3)
    test_facet_ell_dict3 = test_facet_ell_union3.asDict()
    test_facet_ell_union4 = _hpmc.mfellipsoid_params(faceted_ell_union_args4)
    test_facet_ell_dict4 = test_facet_ell_union4.asDict()
    test_facet_ell_union5 = _hpmc.mfellipsoid_params(faceted_ell_union_args5)
    test_facet_ell_dict5 = test_facet_ell_union5.asDict()
    test_facet_ell_union6 = _hpmc.mfellipsoid_params(faceted_ell_union_args6)
    test_facet_ell_dict6 = test_facet_ell_union6.asDict()
    test_facet_ell_union7 = _hpmc.mfellipsoid_params(faceted_ell_union_args7)
    test_facet_ell_dict7 = test_facet_ell_union7.asDict()
    test_facet_ell_union8 = _hpmc.mfellipsoid_params(faceted_ell_union_args8)
    test_facet_ell_dict8 = test_facet_ell_union8.asDict()
    test_facet_ell_union9 = _hpmc.mfellipsoid_params(faceted_ell_union_args9)
    test_facet_ell_dict9 = test_facet_ell_union9.asDict()
    test_facet_ell_union10 = _hpmc.mfellipsoid_params(faceted_ell_union_args10)
    test_facet_ell_dict10 = test_facet_ell_union10.asDict()
    test_facet_ell_union11 = _hpmc.mfellipsoid_params(faceted_ell_union_args11)
    test_facet_ell_dict11 = test_facet_ell_union11.asDict()
    test_facet_ell_union12 = _hpmc.mfellipsoid_params(faceted_ell_union_args12)
    test_facet_ell_dict12 = test_facet_ell_union12.asDict()
    test_facet_ell_union13 = _hpmc.mfellipsoid_params(faceted_ell_union_args13)
    test_facet_ell_dict13 = test_facet_ell_union13.asDict()
    test_facet_ell_union14 = _hpmc.mfellipsoid_params(faceted_ell_union_args14)
    test_facet_ell_dict14 = test_facet_ell_union14.asDict()
    test_facet_ell_union15 = _hpmc.mfellipsoid_params(faceted_ell_union_args15)
    test_facet_ell_dict15 = test_facet_ell_union15.asDict()
    test_facet_ell_union16 = _hpmc.mfellipsoid_params(faceted_ell_union_args16)
    test_facet_ell_dict16 = test_facet_ell_union16.asDict()
    test_facet_ell_union17 = _hpmc.mfellipsoid_params(faceted_ell_union_args17)
    test_facet_ell_dict17 = test_facet_ell_union17.asDict()
    test_facet_ell_union18 = _hpmc.mfellipsoid_params(faceted_ell_union_args18)
    test_facet_ell_dict18 = test_facet_ell_union18.asDict()
    test_facet_ell_union19 = _hpmc.mfellipsoid_params(faceted_ell_union_args19)
    test_facet_ell_dict19 = test_facet_ell_union19.asDict()
    test_facet_ell_union20 = _hpmc.mfellipsoid_params(faceted_ell_union_args20)
    test_facet_ell_dict20 = test_facet_ell_union20.asDict()
    test_facet_ell_union21 = _hpmc.mfellipsoid_params(faceted_ell_union_args21)
    test_facet_ell_dict21 = test_facet_ell_union21.asDict()
    test_facet_ell_union22 = _hpmc.mfellipsoid_params(faceted_ell_union_args22)
    test_facet_ell_dict22 = test_facet_ell_union22.asDict()

    assert test_facet_ell_dict1 == faceted_ell_union_args1
    assert test_facet_ell_dict2 == faceted_ell_union_args2
    assert test_facet_ell_dict3 == faceted_ell_union_args3
    assert test_facet_ell_dict4 == faceted_ell_union_args4
    assert test_facet_ell_dict5 == faceted_ell_union_args5
    assert test_facet_ell_dict6 == faceted_ell_union_args6
    assert test_facet_ell_dict7 == faceted_ell_union_args7
    assert test_facet_ell_dict8 == faceted_ell_union_args8
    assert test_facet_ell_dict9 == faceted_ell_union_args9
    assert test_facet_ell_dict10 == faceted_ell_union_args10
    assert test_facet_ell_dict11 == faceted_ell_union_args11
    assert test_facet_ell_dict12 == faceted_ell_union_args12
    assert test_facet_ell_dict13 == faceted_ell_union_args13
    assert test_facet_ell_dict14 == faceted_ell_union_args14
    assert test_facet_ell_dict15 == faceted_ell_union_args15
    assert test_facet_ell_dict16 == faceted_ell_union_args16
    assert test_facet_ell_dict17 == faceted_ell_union_args17
    assert test_facet_ell_dict18 == faceted_ell_union_args18
    assert test_facet_ell_dict19 == faceted_ell_union_args19
    assert test_facet_ell_dict20 == faceted_ell_union_args20
    assert test_facet_ell_dict21 == faceted_ell_union_args21
    assert test_facet_ell_dict22 == faceted_ell_union_args22


def test_shape_params():

    mc = hoomd.hpmc.integrate.FacetedEllipsoidUnion(23456)

    mc.shape['A'] = dict()
    assert mc.shape['A']['shapes'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['positions'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['orientations'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['overlap'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is False

    mc.shape['A'] = dict(shapes=faceted_ell_union_args1['shapes'])
    assert mc.shape['A']['shapes'] == faceted_ell_union_args1['shapes']
    assert mc.shape['A']['positions'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['orientations'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['overlap'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is False

    mc.shape['A'] = dict(positions=faceted_ell_union_args2['positions'],
                         ignore_statistics=True)
    assert mc.shape['A']['shapes'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['positions'] == faceted_ell_union_args2['positions']
    assert mc.shape['A']['orientations'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['overlap'] == hoomd.typeconverter.RequiredArg
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is True

    mc.shape['A'] = faceted_ell_union_args3
    test_args = faceted_ell_union_args3
    assert mc.shape['A']['shapes'] == test_args['shapes']
    assert mc.shape['A']['positions'] == test_args['positions']
    assert mc.shape['A']['orientations'] == test_args['orientations']
    assert mc.shape['A']['overlap'] == test_args['overlap']
    assert mc.shape['A']['capacity'] == test_args['capacity']
    assert mc.shape['A']['ignore_statistics'] == test_args['ignore_statistics']


def test_shape_params_attached(device, dummy_simulation_factory):

    mc = hoomd.hpmc.integrate.FacetedEllipsoidUnion(23456)
    mc.shape['A'] = faceted_ell_union_args1
    mc.shape['B'] = faceted_ell_union_args2
    mc.shape['C'] = faceted_ell_union_args3
    mc.shape['D'] = faceted_ell_union_args4
    mc.shape['E'] = faceted_ell_union_args5
    mc.shape['F'] = faceted_ell_union_args6
    mc.shape['G'] = faceted_ell_union_args7
    mc.shape['H'] = faceted_ell_union_args8
    mc.shape['I'] = faceted_ell_union_args9
    mc.shape['J'] = faceted_ell_union_args10
    mc.shape['K'] = faceted_ell_union_args11

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D', 'E', 'F',
                                                   'G', 'H', 'I', 'J', 'K'])
    sim.operations.add(mc)
    sim.operations.schedule()

    test_args = faceted_ell_union_args1
    assert mc.shape['A']['shapes'] == test_args['shapes']
    assert mc.shape['A']['positions'] == test_args['positions']
    assert mc.shape['A']['orientations'] == test_args['orientations']
    assert mc.shape['A']['overlap'] == test_args['overlap']
    assert mc.shape['A']['capacity'] == test_args['capacity']
    assert mc.shape['A']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args2
    assert mc.shape['B']['shapes'] == test_args['shapes']
    assert mc.shape['B']['positions'] == test_args['positions']
    assert mc.shape['B']['orientations'] == test_args['orientations']
    assert mc.shape['B']['overlap'] == test_args['overlap']
    assert mc.shape['B']['capacity'] == test_args['capacity']
    assert mc.shape['B']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args3
    assert mc.shape['C']['shapes'] == test_args['shapes']
    assert mc.shape['C']['positions'] == test_args['positions']
    assert mc.shape['C']['orientations'] == test_args['orientations']
    assert mc.shape['C']['overlap'] == test_args['overlap']
    assert mc.shape['C']['capacity'] == test_args['capacity']
    assert mc.shape['C']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args4
    assert mc.shape['D']['shapes'] == test_args['shapes']
    assert mc.shape['D']['positions'] == test_args['positions']
    assert mc.shape['D']['orientations'] == test_args['orientations']
    assert mc.shape['D']['overlap'] == test_args['overlap']
    assert mc.shape['D']['capacity'] == test_args['capacity']
    assert mc.shape['D']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args5
    assert mc.shape['E']['shapes'] == test_args['shapes']
    assert mc.shape['E']['positions'] == test_args['positions']
    assert mc.shape['E']['orientations'] == test_args['orientations']
    assert mc.shape['E']['overlap'] == test_args['overlap']
    assert mc.shape['E']['capacity'] == test_args['capacity']
    assert mc.shape['E']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args6
    assert mc.shape['F']['shapes'] == test_args['shapes']
    assert mc.shape['F']['positions'] == test_args['positions']
    assert mc.shape['F']['orientations'] == test_args['orientations']
    assert mc.shape['F']['overlap'] == test_args['overlap']
    assert mc.shape['F']['capacity'] == test_args['capacity']
    assert mc.shape['F']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args7
    assert mc.shape['G']['shapes'] == test_args['shapes']
    assert mc.shape['G']['positions'] == test_args['positions']
    assert mc.shape['G']['orientations'] == test_args['orientations']
    assert mc.shape['G']['overlap'] == test_args['overlap']
    assert mc.shape['G']['capacity'] == test_args['capacity']
    assert mc.shape['G']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args8
    assert mc.shape['H']['shapes'] == test_args['shapes']
    assert mc.shape['H']['positions'] == test_args['positions']
    assert mc.shape['H']['orientations'] == test_args['orientations']
    assert mc.shape['H']['overlap'] == test_args['overlap']
    assert mc.shape['H']['capacity'] == test_args['capacity']
    assert mc.shape['H']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args9
    assert mc.shape['I']['shapes'] == test_args['shapes']
    assert mc.shape['I']['positions'] == test_args['positions']
    assert mc.shape['I']['orientations'] == test_args['orientations']
    assert mc.shape['I']['overlap'] == test_args['overlap']
    assert mc.shape['I']['capacity'] == test_args['capacity']
    assert mc.shape['I']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args10
    assert mc.shape['J']['shapes'] == test_args['shapes']
    assert mc.shape['J']['positions'] == test_args['positions']
    assert mc.shape['J']['orientations'] == test_args['orientations']
    assert mc.shape['J']['overlap'] == test_args['overlap']
    assert mc.shape['J']['capacity'] == test_args['capacity']
    assert mc.shape['J']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = faceted_ell_union_args11
    assert mc.shape['K']['shapes'] == test_args['shapes']
    assert mc.shape['K']['positions'] == test_args['positions']
    assert mc.shape['K']['orientations'] == test_args['orientations']
    assert mc.shape['K']['overlap'] == test_args['overlap']
    assert mc.shape['K']['capacity'] == test_args['capacity']
    assert mc.shape['K']['ignore_statistics'] == test_args['ignore_statistics']

    faceted_ell_union_args1_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args2_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args3_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args4_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args5_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args6_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args7_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args8_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args9_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args10_invalid = copy.deepcopy(faceted_ell_union_args1)
    faceted_ell_union_args11_invalid = copy.deepcopy(faceted_ell_union_args1)

    faceted_ell_union_args1_invalid['shapes'] = 'invalid'
    faceted_ell_union_args2_invalid['shapes'] = 1
    faceted_ell_union_args3_invalid['shapes'] = [1, 2, 3]
    faceted_ell_union_args4_invalid['orientations'] = 'invalid'
    faceted_ell_union_args5_invalid['orientations'] = 1
    faceted_ell_union_args6_invalid['positions'] = 1
    faceted_ell_union_args7_invalid['positions'] = [1, 2, 3]
    faceted_ell_union_args8_invalid['positions'] = 'invalid'
    faceted_ell_union_args9_invalid['overlap'] = 'invalid'
    faceted_ell_union_args10_invalid['capacity'] = 'invalid'
    faceted_ell_union_args11_invalid['capacity'] = [1, 2, 3]

    # check for errors on invalid input
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args1_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args2_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = faceted_ell_union_args3_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args4_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args5_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args6_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = faceted_ell_union_args7_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args8_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args9_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args10_invalid

    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        mc.shape['A'] = faceted_ell_union_args11_invalid
