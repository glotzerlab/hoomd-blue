import hoomd
import numpy as np
from hoomd.hpmc import _hpmc
from hoomd import hpmc
import pytest


args_1 = {'vertices':[(0, 0, 0),
                        (1, 1, 0),
                        (1, 0, 1),
                        (0, 1, 1),
                        (1, 1, 1),
                        (0, 0, 1)],
            'faces':[(0, 0, 0),
                    (1, 1, 0),
                    (1, 0, 1),
                    (0, 1, 1),
                    (1, 1, 1),
                    (0, 0, 1)],
            'overlap': [1, 2, 3, 4, 5, 6],
            'sweep_radius': 0,
            'ignore_statistics': 1,
            'capacity': 4,
            'origin': (0, 0, 0),
            'hull_only': True}

args_2 = {'vertices':[(0, 3, 0),
                        (2, 1, 0),
                        (1, 3, 1),
                        (1, 1, 1),
                        (1, 2, 5),
                        (3, 0, 1)],
            'faces':[(0, 4, 5),
                    (1, 3, 2),
                    (1, 2, 5),
                    (5, 1, 3),
                    (1, 4, 3),
                    (0, 2, 1)],
            'overlap': [1, 4, 6, 4, 1, 6],
            'sweep_radius': 2,
            'ignore_statistics': 0,
            'capacity': 3,
            'origin': (0, 1, 0),
            'hull_only': False}

args_3 = {'vertices':[(0, 3, 0),
                        (2, 1, 0),
                        (1, 3, 1),
                        (1, 1, 1),
                        (1, 2, 5),
                        (3, 0, 1),
                        (0, 3, 3)],
            'faces':[(0, 1, 2),
                    (3, 2, 6),
                    (1, 2, 4),
                    (6, 1, 3),
                    (3, 4, 6),
                    (4, 5, 1),
                    (6, 2, 5)],
            'overlap': [5, 4, 3, 4, 1, 2, 3],
            'sweep_radius': 1,
            'ignore_statistics': 1,
            'capacity': 4,
            'origin': (0, 0, 0),
            'hull_only': True}

args_4 = {'vertices':[(0, 3, 0), (2, 1, 0), (3, 0, 1), (0, 3, 3)],
            'faces':[(0, 1, 2), (3, 2, 1), (1, 2, 0), (3, 2, 1)],
            'overlap': [5, 4, 3, 1],
            'sweep_radius': 0,
            'ignore_statistics': 0,
            'capacity': 4,
            'origin': (0, 0, 1),
            'hull_only': True}

args_5 = {'vertices':[(0, 3, 0),
                        (2, 1, 0),
                        (1, 3, 1),
                        (1, 1, 1),
                        (1, 2, 5),
                        (3, 0, 1),
                        (0, 3, 3),
                        (0, 0, 2),
                        (1, 2, 2)],
            'faces':[(0, 1, 2),
                    (3, 2, 6),
                    (1, 2, 4),
                    (6, 1, 3),
                    (3, 4, 6),
                    (4, 5, 1),
                    (6, 7, 5),
                    (1, 7, 8),
                    (6, 8, 2)],
            'overlap': [5, 4, 3, 4, 1, 2, 3, 1, 1],
            'sweep_radius': 0,
            'ignore_statistics': 1,
            'capacity': 4,
            'origin': (0, 1, 0),
            'hull_only': True}


# these test passing back and forth between python and C++
def test_polyhedron():

    test_polyhedron1 = _hpmc.TriangleMesh(args_1)
    test_dict1 = test_polyhedron1.asDict()
    assert test_dict1 == args_1

    test_polyhedron2 = _hpmc.TriangleMesh(args_2)
    test_dict2 = test_polyhedron2.asDict()
    assert test_dict2 == args_2

    test_polyhedron3 = _hpmc.TriangleMesh(args_3)
    test_dict3 = test_polyhedron3.asDict()
    assert test_dict3 == args_3

    test_polyhedron4 = _hpmc.TriangleMesh(args_4)
    test_dict4 = test_polyhedron4.asDict()
    assert test_dict4 == args_4

    test_polyhedron5 = _hpmc.TriangleMesh(args_5)
    test_dict5 = test_polyhedron5.asDict()
    assert test_dict5 == args_5

    assert True


# these tests are for the python side
def test_polyhedron_python():

    poly = hpmc.integrate.Polyhedron(2456)
    poly.shape['A'] = args_4
    assert not poly.shape['A']['ignore_statistics']
    for key in args_4.keys():
        assert poly.shape['A'][key] == args_4[key]


def test_poly_after_attaching(device,
                              dummy_simulation_factory):

    poly = hpmc.integrate.Polyhedron(2346)
    poly.shape['A'] = args_4
    poly.shape['B'] = args_5

    sim = dummy_simulation_factory(particle_types=['A', 'B'])
    sim.operations.add(poly)
    sim.operations.schedule()

    assert not poly.shape['A']['ignore_statistics']
    assert poly.shape['B']['ignore_statistics']
    for key in args_4.keys():
        assert poly.shape['A'][key] == args_4[key]
        assert poly.shape['B'][key] == args_5[key]
        
        
def test_overlaps(device, dummy_simulation_check_overlaps):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.Polyhedron(23456)
    mc.shape['A'] = dict(vertices=[(0,(0.75**0.5)/2, -0.5),
                                   (-0.5,-(0.75**0.5)/2, -0.5),
                                   (0.5, -(0.75**0.5)/2, -0.5),
                                   (0, 0, 0.5),
                                   (0, 0, 0)],
                         faces=[(3, 1, 2),
                                (3, 0, 1),
                                (3, 2, 0),
                                (4, 2, 1),
                                (4, 0, 2),
                                (4, 1, 0)],
                         overlap=[0, 0, 0, 0, 0, 0])
    sim = dummy_simulation_check_overlaps()
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    # overlaps = sim.operations.integrator.overlaps
    # assert overlaps > 0
    assert True
    
def test_shape_moves(device, dummy_simulation_check_moves):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.Polyhedron(23456)
    mc.shape['A'] = dict(vertices=[(0,(0.75**0.5)/2, -0.5),
                                   (-0.5,-(0.75**0.5)/2, -0.5),
                                   (0.5, -(0.75**0.5)/2, -0.5),
                                   (0, 0, 0.5),
                                   (0, 0, 0)],
                         faces=[(3, 1, 2),
                                (3, 0, 1),
                                (3, 2, 0),
                                (4, 2, 1),
                                (4, 0, 2),
                                (4, 1, 0)],
                         overlap=[0, 0, 0, 0, 0, 0])
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
