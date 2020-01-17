import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
import pytest
import copy

sph_args_1 = {'diameter': 1, 'orientable': 0, 'ignore_statistics': 1}
sph_args_2 = {'diameter': 9, 'orientable': 1, 'ignore_statistics': 1}
sph_args_3 = {'diameter': 4, 'orientable': 0, 'ignore_statistics': 0}

sph_union_args1 = {'shapes': [sph_args_1, sph_args_2],
                   'positions': [(0, 0, 0), (0, 0, 1)],
                   'orientations': [(1, 0, 0, 0), (1, 1, 0, 0)],
                   'overlap': [1, 1],
                   'capacity': 4,
                   'ignore_statistics': 1}

sph_union_args2 = {'shapes': [sph_args_1, sph_args_3],
                   'positions': [(1, 0, 0), (0, 0, 1)],
                   'orientations': [(1, 1, 0, 0), (1, 0, 0, 0)],
                   'overlap': [1, 2],
                   'capacity': 3,
                   'ignore_statistics': 1}

sph_union_args3 = {'shapes': [sph_args_3, sph_args_2],
                   'positions': [(1, 1, 0), (1, 0, 1)],
                   'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                   'overlap': [1, 0],
                   'capacity': 2,
                   'ignore_statistics': 0}

sph_union_args4 = {'shapes': [sph_args_1, sph_args_2, sph_args_3],
                   'positions': [(0, 0, 0), (0, 1, 1), (1, 1, 1)],
                   'orientations': [(1, 1, 1, 0), (1, 1, 0, 0), (1, 0, 0, 1)],
                   'overlap': [1, 1, 1],
                   'capacity': 4,
                   'ignore_statistics': 1}


def test_dict_conversions():

    test_sph_union1 = _hpmc.SphereUnionParams(sph_union_args1)
    test_sph_dict1 = test_sph_union1.asDict()
    assert test_sph_dict1 == sph_union_args1

    test_sph_union2 = _hpmc.SphereUnionParams(sph_union_args2)
    test_sph_dict2 = test_sph_union2.asDict()
    assert test_sph_dict2 == sph_union_args2

    test_sph_union3 = _hpmc.SphereUnionParams(sph_union_args3)
    test_sph_dict3 = test_sph_union3.asDict()
    assert test_sph_dict3 == sph_union_args3

    test_sph_union4 = _hpmc.SphereUnionParams(sph_union_args4)
    test_sph_dict4 = test_sph_union4.asDict()
    assert test_sph_dict4 == sph_union_args4


def test_shape_params():

    mc = hoomd.hpmc.integrate.SphereUnion(23456)

    mc.shape['A'] = dict()
    print(mc.shape['A'])
    assert mc.shape['A']['shapes'] is None
    assert mc.shape['A']['positions'] is None
    assert mc.shape['A']['orientations'] is None
    assert mc.shape['A']['overlap'] == 1
    assert mc.shape['A']['capacity'] == 4
    assert mc.shape['A']['ignore_statistics'] is False


def test_shape_params_attached(device, dummy_simulation_factory):

    mc = hoomd.hpmc.integrate.SphereUnion(23456)
    mc.shape['A'] = sph_union_args1
    mc.shape['B'] = sph_union_args2
    mc.shape['C'] = sph_union_args3
    mc.shape['D'] = sph_union_args4

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D'])
    sim.operations.add(mc)
    sim.operations.schedule()

    test_args = sph_union_args1
    assert mc.shape['A']['shapes'] == test_args['shapes']
    assert mc.shape['A']['positions'] == test_args['positions']
    assert mc.shape['A']['orientations'] == test_args['orientations']
    assert mc.shape['A']['overlap'] == test_args['overlap']
    assert mc.shape['A']['capacity'] == test_args['capacity']
    assert mc.shape['A']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = sph_union_args2
    assert mc.shape['B']['shapes'] == test_args['shapes']
    assert mc.shape['B']['positions'] == test_args['positions']
    assert mc.shape['B']['orientations'] == test_args['orientations']
    assert mc.shape['B']['overlap'] == test_args['overlap']
    assert mc.shape['B']['capacity'] == test_args['capacity']
    assert mc.shape['B']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = sph_union_args3
    assert mc.shape['C']['shapes'] == test_args['shapes']
    assert mc.shape['C']['positions'] == test_args['positions']
    assert mc.shape['C']['orientations'] == test_args['orientations']
    assert mc.shape['C']['overlap'] == test_args['overlap']
    assert mc.shape['C']['capacity'] == test_args['capacity']
    assert mc.shape['C']['ignore_statistics'] == test_args['ignore_statistics']

    test_args = sph_union_args4
    assert mc.shape['D']['shapes'] == test_args['shapes']
    assert mc.shape['D']['positions'] == test_args['positions']
    assert mc.shape['D']['orientations'] == test_args['orientations']
    assert mc.shape['D']['overlap'] == test_args['overlap']
    assert mc.shape['D']['capacity'] == test_args['capacity']
    assert mc.shape['D']['ignore_statistics'] == test_args['ignore_statistics']

    sph_union_args1_invalid = copy.deepcopy(sph_union_args1)
    sph_union_args2_invalid = copy.deepcopy(sph_union_args2)
    sph_union_args3_invalid = copy.deepcopy(sph_union_args3)
    sph_union_args4_invalid = copy.deepcopy(sph_union_args4)
    sph_union_args1_invalid['shapes'] = 'invalid'
    sph_union_args2_invalid['shapes'] = 1
    sph_union_args3_invalid['shapes'] = [1, 2, 3]
    sph_union_args4_invalid['orientations'] = 'invalid'

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args1_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args2_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args3_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args4_invalid

    sph_union_args1_invalid = copy.deepcopy(sph_union_args1)
    sph_union_args2_invalid = copy.deepcopy(sph_union_args2)
    sph_union_args3_invalid = copy.deepcopy(sph_union_args3)
    sph_union_args4_invalid = copy.deepcopy(sph_union_args4)
    sph_union_args1_invalid['orientations'] = 1
    sph_union_args2_invalid['positions'] = 1
    sph_union_args3_invalid['positions'] = [1, 2, 3]
    sph_union_args4_invalid['positions'] = 'invalid'

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args1_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args2_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args3_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args4_invalid

    sph_union_args1_invalid['overlap'] = 'invalid'
    sph_union_args2_invalid['capacity'] = 'invalid'
    sph_union_args3_invalid['capacity'] = [1, 2, 3]
    sph_union_args4_invalid['ignore_statistics'] = 'invalid'

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args1_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = sph_union_args2_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args3_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = sph_union_args4_invalid
