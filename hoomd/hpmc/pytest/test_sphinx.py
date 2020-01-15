import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc
import pytest
import copy
import numpy

args_1 = {'diameters':[1, 4, 2, 8, 5, 9],
            'centers':[(0, 0, 0),
                        (1, 1, 1),
                        (1, 0, 1),
                        (0, 1, 1),
                        (1, 1, 0),
                        (0, 0, 1)],
            'ignore_statistics':1}

args_2 = {'diameters':[5, 2, 4, 5, 1, 2],
            'centers':[(0, 2, 0),
                        (1, 4, 1),
                        (3, 0, 1),
                        (3, 1, 1),
                        (1, 4, 0),
                        (2, 2, 1)],
            'ignore_statistics':0}

args_3 = {'diameters':[1, 2, 2, 3, 4, 9, 3, 2],
            'centers':[(0, 0, 0),
                        (1, 1, 1),
                        (1, 0, 1),
                        (0, 1, 1),
                        (1, 1, 0),
                        (0, 0, 1),
                        (2, 2, 1),
                        (3, 5, 3)],
            'ignore_statistics':1}

args_4 = {'diameters':[1, 4, 2, 8, 5],
            'centers':[(0, 2, 0), (1, 4, 1), (3, 0, 1), (3, 1, 1), (1, 4, 0)],
            'ignore_statistics':0}


def test_sphinx():

    test_sphinx1 = hpmc.SphinxParams(args_1)
    test_dict1 = test_sphinx1.asDict()
    assert test_dict1 == args_1

    test_sphinx2 = hpmc.SphinxParams(args_2)
    test_dict2 = test_sphinx2.asDict()
    assert test_dict2 == args_2

    test_sphinx3 = hpmc.SphinxParams(args_3)
    test_dict3 = test_sphinx3.asDict()
    assert test_dict3 == args_3

    test_sphinx4 = hpmc.SphinxParams(args_4)
    test_dict4 = test_sphinx4.asDict()
    assert test_dict4 == args_4


def test_shape_params():

    mc = hoomd.hpmc.integrate.Sphinx(23456)

    mc.shape['A'] = dict()
    assert mc.shape['A']['diameters'] is None
    assert mc.shape['A']['centers'] is None
    assert mc.shape['A']['ignore_statistics'] is False

    mc.shape['B'] = dict(diameters=[1, 4, 2, 8, 5, 9],
                         centers=[(0, 2, 0),
                                  (1, 4, 1),
                                  (3, 0, 1),
                                  (3, 1, 1),
                                  (1, 4, 0),
                                  (2, 2, 1)],
                         ignore_statistics=True)
    assert mc.shape['B']['diameters'] == [1, 4, 2, 8, 5, 9]
    assert mc.shape['B']['centers'] == [(0, 2, 0),
                                        (1, 4, 1),
                                        (3, 0, 1),
                                        (3, 1, 1),
                                        (1, 4, 0),
                                        (2, 2, 1)]
    assert mc.shape['B']['ignore_statistics'] is True

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


def test_shape_params_attached(device, dummy_simulation_factory):

    mc = hoomd.hpmc.integrate.Sphinx(23456)

    mc.shape['A'] = args_1
    mc.shape['B'] = args_2
    mc.shape['C'] = args_3
    mc.shape['D'] = args_4

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D'])
    sim.operations.add(mc)
    sim.operations.schedule()

    assert mc.shape['A']['ignore_statistics']
    assert not mc.shape['B']['ignore_statistics']
    for key in args_1.keys():
        assert mc.shape['A'][key] == args_1[key]
        assert mc.shape['B'][key] == args_2[key]
        assert mc.shape['C'][key] == args_3[key]
        assert mc.shape['D'][key] == args_4[key]


    args_1_invalid = copy.deepcopy(args_1)
    args_2_invalid = copy.deepcopy(args_2)
    args_3_invalid = copy.deepcopy(args_3)
    args_4_invalid = copy.deepcopy(args_4)
    args_1_invalid['diameters'] = 'invalid'
    args_2_invalid['diameters'] = 1
    args_3_invalid['diameters'] = [(0, 0, 0),
                           (1, 1, 1),
                           (1, 0, 1),
                           (0, 1, 1),
                           (1, 1, 0),
                           (0, 0, 1),
                           (2, 2, 1),
                           (3, 5, 3)]
    args_4_invalid['ignore_statistics'] = 'invalid'

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_1_invalid)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_2_invalid)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_3_invalid)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_4_invalid)

    args_1_invalid = copy.deepcopy(args_1)
    args_2_invalid = copy.deepcopy(args_2)
    args_3_invalid = copy.deepcopy(args_3)
    args_1_invalid['centers'] = 'invalid'
    args_2_invalid['centers'] = 1
    args_3_invalid['centers'] = [1, 2, 3, 4, 5, 6, 7, 8]

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(args_1_invalid)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_2_invalid)

    with pytest.raises(TypeError):
        mc.shape['A'] = dict(args_3_invalid)
        
    with pytest.raises(ValueError):
        mc.shape['A'] = dict(vertices=[(0,(0.75**0.5)/2, -0.5),
                                       (-0.5,-(0.75**0.5)/2, -0.5),
                                       (0.5, -(0.75**0.5)/2, -0.5),
                                       (0, 0, 0.5)])

def test_overlaps(device, dummy_simulation_check_overlaps):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.Sphinx(23456)
    mc.shape['A'] = dict(centers=[(0,0,0),(0.5,0,0)], diameters=[1,.001])
    
    sim = dummy_simulation_check_overlaps()
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    overlaps = sim.operations.integrator.overlaps
    assert overlaps > 0
    # assert True
    
def test_shape_moves(device, dummy_simulation_check_moves):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.Sphinx(23456)
    mc.shape['A'] = dict(centers=[(0,0,0),(0.5,0,0)], diameters=[1,.1])
    sim = dummy_simulation_check_moves()
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    accepted_and_rejected_rotations = sum(sim.operations.integrator.rotate_moves)
    #print(sim.operations.integrator.rotate_moves)
    #print(sim.operations.integrator.translate_moves)
    #assert accepted_and_rejected_rotations > 0
    accepted_and_rejected_translations = sum(sim.operations.integrator.translate_moves)
    assert accepted_and_rejected_translations > 0
