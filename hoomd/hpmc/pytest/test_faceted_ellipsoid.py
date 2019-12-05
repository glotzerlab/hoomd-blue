import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc
import pytest
import numpy as np

@pytest.fixture(scope='session')
def dummy_integrator_args():
    args_1 = {"normals": [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1)],
              "offsets": [1, 3, 2, 6, 3, 1],
              "a": 3,
              "b": 4,
              "c": 1,
              "vertices": [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (1, 1, 0)],
              "origin": (0, 0, 0),
              "ignore_statistics":1}

    args_2 = {"normals": [(0, 0, 0), (2, 1, 1), (1, 3, 3), (5, 1, 1), (1, 3, 0), (1, 2, 2)],
              "offsets": [1, 3, 3, 2, 3, 1],
              "a": 2,
              "b": 1,
              "c": 3,
              "vertices": [(1, 0, 0), (1, 1, 0), (1, 2, 1), (0, 1, 1), (1, 1, 2), (0, 0, 1)],
              "origin": (0, 0, 1),
              "ignore_statistics":0}

    args_3 = {"normals": [(0, 0, 2), (0, 1, 1), (1, 3, 5), (0, 1, 6)],
              "offsets": [6, 2, 2, 5],
              "a": 1,
              "b": 6,
              "c": 6,
              "vertices": [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)],
              "origin": (0, 1, 0),
              "ignore_statistics":1}

    args_4 = {"normals": [(0, 0, 2), (2, 2, 0), (3, 1, 1), (4, 1, 1), (1, 2, 0), (3, 3, 1), (1, 2, 1), (3, 3, 2)],
              "offsets": [5, 3, 3, 4, 3, 4, 2, 2],
              "a": 2,
              "b": 2,
              "c": 4,
              "vertices": [(0, 1, 0), (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1), (0, 0, 1), (0, 0, 1)],
              "origin": (1, 0, 0),
              "ignore_statistics":0}

    args_5 = {"normals": [(0, 0, 1), (0, 4, 0), (2, 0, 1), (0, 3, 1), (4, 1, 0), (2, 2, 1), (1, 3, 1), (1, 9, 0), (2, 2, 2)],
              "offsets": [5, 4, 2, 2, 7, 3, 1, 4, 1],
              "a": 6,
              "b": 1,
              "c": 1,
              "vertices": [(0, 10, 3), (3, 2, 1), (1, 2, 1), (0, 1, 1), (1, 1, 0), (5, 0, 1), (0, 10, 1), (9, 5, 1), (0, 0, 1)],
              "origin": (0, 0, 0),
              "ignore_statistics":1}

    return args_1, args_2, args_3, args_4, args_5

def test_faceted_ellipsoid(dummy_integrator_args):

    args_1, args_2, args_3, args_4, args_5 = dummy_integrator_args

    test_faceted_ellipsoid1 = hpmc.FacetedEllipsoidParams(args_1)
    test_dict1 = test_faceted_ellipsoid1.asDict()
    assert test_dict1 == args_1

    test_faceted_ellipsoid2 = hpmc.FacetedEllipsoidParams(args_2)
    test_dict2 = test_faceted_ellipsoid2.asDict()
    assert test_dict2 == args_2

    test_faceted_ellipsoid3 = hpmc.FacetedEllipsoidParams(args_3)
    test_dict3 = test_faceted_ellipsoid3.asDict()
    assert test_dict3 == args_3

    test_faceted_ellipsoid4 = hpmc.FacetedEllipsoidParams(args_4)
    test_dict4 = test_faceted_ellipsoid4.asDict()
    assert test_dict4 == args_4

    test_faceted_ellipsoid5 = hpmc.FacetedEllipsoidParams(args_5)
    test_dict5 = test_faceted_ellipsoid5.asDict()
    assert test_dict5 == args_5

def test_shape_params(dummy_integrator_args):

    mc = hoomd.hpmc.integrate.FacetedEllipsoid(23456)

    args_1, args_2, args_3, args_4, args_5 = dummy_integrator_args

    mc.shape['A'] = dict()
    assert mc.shape['A']['normals'] is None
    assert mc.shape['A']['offsets'] is None
    assert mc.shape['A']['vertices'] is None
    assert mc.shape['A']['origin'] is None
    assert mc.shape['A']['a'] is None
    assert mc.shape['A']['b'] is None
    assert mc.shape['A']['c'] is None
    assert mc.shape['A']['ignore_statistics'] is False

    mc.shape['B'] = dict(a=2.5, b=1, c=3)
    assert mc.shape['B']['normals'] is None
    assert mc.shape['B']['offsets'] is None
    assert mc.shape['B']['vertices'] is None
    assert mc.shape['B']['origin'] is None
    assert mc.shape['B']['a'] == 2.5
    assert mc.shape['B']['b'] == 1
    assert mc.shape['B']['c'] == 3
    assert mc.shape['B']['ignore_statistics'] is False

    mc.shape['C'] = args_1
    for key in args_1.keys():
        assert mc.shape['C'][key] == args_1[key]

    mc.shape['D'] = args_2
    for key in args_2.keys():
        assert mc.shape['D'][key] == args_2[key]

    mc.shape['E'] = args_3
    for key in args_3.keys():
        assert mc.shape['E'][key] == args_3[key]

    mc.shape['F'] = args_4
    for key in args_4.keys():
        assert mc.shape['F'][key] == args_4[key]

    mc.shape['G'] = args_5
    for key in args_5.keys():
        assert mc.shape['G'][key] == args_5[key]



def test_shape_params_attached(device, dummy_simulation_factory, dummy_integrator_args):

    args_1, args_2, args_3, args_4, args_5 = dummy_integrator_args

    mc = hoomd.hpmc.integrate.FacetedEllipsoid(23456)

    mc.shape['A'] = args_1
    mc.shape['B'] = args_2
    mc.shape['C'] = args_3
    mc.shape['D'] = args_4
    mc.shape['E'] = args_5

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D', 'E'])
    sim.operations.add(mc)
    sim.operations.schedule()

    assert mc.shape['A']['ignore_statistics']
    assert not mc.shape['B']['ignore_statistics']
    for key in args_1.keys():
        assert mc.shape['A'][key] == args_1[key]
        assert mc.shape['B'][key] == args_2[key]
        assert mc.shape['C'][key] == args_3[key]
        assert mc.shape['D'][key] == args_4[key]
        assert mc.shape['E'][key] == args_5[key]

    args_1['normals'] = 'invalid'
    args_2['normals'] = 1
    args_3['normals'] = [1, 2, 3, 4]
    args_4['origin'] = 1
    args_5['origin'] = 'invalid'

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_1)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_2)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_3)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_4)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_5)


    args_1, args_2, args_3, args_4, args_5 = dummy_integrator_args

    args_1['offsets'] = 'invalid'
    args_2['offsets'] = 1
    args_3['offsets'] = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 1,)]
    args_4['ignore_statistics'] = 'invalid'

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_1)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_2)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_3)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_4)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_5)


    args_1, args_2, args_3, args_4, args_5 = dummy_integrator_args

    args_1['vertices'] = 'invalid'
    args_2['vertices'] = 1
    args_3['vertices'] = [1, 2, 3, 4]
    args_4['a'] = [1, 2, 3]
    args_5['b'] = 'invalid'

     # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_1)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_2)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_3)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_4)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_5)

