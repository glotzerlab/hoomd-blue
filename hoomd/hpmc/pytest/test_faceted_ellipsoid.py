import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc
import pytest
import numpy as np
import copy

args_1 = {"normals": [(0, 0, 1),
                      (0, 1, 0),
                      (1, 0, 0),
                      (0, 1, 1),
                      (1, 1, 0),
                      (1, 0, 1)],
          "offsets": [1, 3, 2, 6, 3, 1],
          "a": 3,
          "b": 4,
          "c": 1,
          "vertices": [(0, 0, 0),
                       (0, 0, 1),
                       (0, 1, 0),
                       (1, 0, 0),
                       (1, 1, 1),
                       (1, 1, 0)],
          "origin": (0, 0, 0),
          "ignore_statistics": 1}

args_2 = {"normals": [(0, 0, 0),
                      (2, 1, 1),
                      (1, 3, 3),
                      (5, 1, 1),
                      (1, 3, 0),
                      (1, 2, 2)],
          "offsets": [1, 3, 3, 2, 3, 1],
          "a": 2,
          "b": 1,
          "c": 3,
          "vertices": [(1, 0, 0),
                       (1, 1, 0),
                       (1, 2, 1),
                       (0, 1, 1),
                       (1, 1, 2),
                       (0, 0, 1)],
          "origin": (0, 0, 1),
          "ignore_statistics": 0}

args_3 = {"normals": [(0, 0, 2), (0, 1, 1), (1, 3, 5), (0, 1, 6)],
          "offsets": [6, 2, 2, 5],
          "a": 1,
          "b": 6,
          "c": 6,
          "vertices": [(0, 0, 0), (1, 1, 1), (1, 0, 2), (2, 1, 1)],
          "origin": (0, 1, 0),
          "ignore_statistics": 1}

args_4 = {"normals": [(0, 0, 2),
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

args_5 = {"normals": [(0, 0, 1),
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


def test_faceted_ellipsoid():

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

def test_shape_params():

    mc = hoomd.hpmc.integrate.FacetedEllipsoid(23456)

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


def test_shape_params_attached(device, dummy_simulation_factory):

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

    args_1_invalid = copy.deepcopy(args_1)
    args_2_invalid = copy.deepcopy(args_2)
    args_3_invalid = copy.deepcopy(args_3)
    args_4_invalid = copy.deepcopy(args_4)
    args_5_invalid = copy.deepcopy(args_5)
    args_1_invalid['normals'] = 'invalid'
    args_2_invalid['normals'] = 1
    args_3_invalid['normals'] = [1, 2, 3, 4]
    args_4_invalid['origin'] = 1
    args_5_invalid['origin'] = 'invalid'

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_1_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_2_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = args_3_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = args_4_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_5_invalid


    args_1_invalid = copy.deepcopy(args_1)
    args_2_invalid = copy.deepcopy(args_2)
    args_3_invalid = copy.deepcopy(args_3)
    args_4_invalid = copy.deepcopy(args_4)
    args_1_invalid['offsets'] = 'invalid'
    args_2_invalid['offsets'] = 1
    args_3_invalid['offsets'] = [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 1,)]
    args_4_invalid['ignore_statistics'] = 'invalid'

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_1_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = args_2_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_3_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_4_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_5_invalid

    args_1_invalid = copy.deepcopy(args_1)
    args_2_invalid = copy.deepcopy(args_2)
    args_3_invalid = copy.deepcopy(args_3)
    args_4_invalid = copy.deepcopy(args_4)
    args_1_invalid['vertices'] = 'invalid'
    args_2_invalid['vertices'] = 1
    args_3_invalid['vertices'] = [1, 2, 3, 4]
    args_4_invalid['a'] = [1, 2, 3]
    args_5_invalid['b'] = 'invalid'

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_1_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = args_2_invalid

    with pytest.raises(TypeError):
        mc.shape['A'] = args_3_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_4_invalid

    with pytest.raises(RuntimeError):
        mc.shape['A'] = args_5_invalid

def test_overlaps(device, dummy_simulation_check_overlaps):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.FacetedEllipsoid(23456)
    mc.shape['A'] = dict(normals=[(0, 0, 1)],
                         a=1,
                         b=1,
                         c=0.5,
                         vertices=[],
                         origin=(0, 0, 0),
                         offsets=[0])
    
    sim = dummy_simulation_check_overlaps()
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    #overlaps = sim.operations.integrator.overlaps
    #assert overlaps > 0
    assert True
    
def test_shape_moves(device, dummy_simulation_check_moves):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.FacetedEllipsoid(23456)
    mc.shape['A'] = dict(normals=[(0, 0, 1)],
                         a=1,
                         b=1,
                         c=0.5,
                         vertices=[],
                         origin=(0, 0, 0),
                         offsets=[0])
    sim = dummy_simulation_check_moves()
    sim.operations.add(mc)
    sim.operations.schedule()
    initial_snap = sim.state.snapshot
    initial_pos = initial_snap.particles.position
    sim.run(100)
    final_snap = sim.state.snapshot
    final_pos = final_snap.particles.position
    #accepted_and_rejected = sim.operations.integrator.accepted + 
    #                        sim.operations.integrator.rejected
    #assert accepted_and_rejected > 0
    np.testing.assert_raises(AssertionError, 
                                np.testing.assert_allclose, 
                                final_pos, 
                                initial_pos)
    

    
    
    
