# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import hoomd.hpmc
import numpy as np
import pytest
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check
from hoomd.hpmc.update import VertexShapeMove, PythonShapeMove, ElasticShapeMove

# def test_before_attaching():
#     verts = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
#                         [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) / 2
#     vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01},
#                                                param_ratio=0.2,
#                                                volume=1.0)
#     move_ratio = 1.0
#     trigger = hoomd.trigger.Periodic(1)
#     nselect = 1
#     nsweeps = 1
#     multi_phase = True
#     num_phase = 2
#     shape_updater = hoomd.hpmc.update.Shape(shape_move=vertex_move,
#                                             move_ratio=move_ratio,
#                                             trigger=trigger,
#                                             nselect=nselect,
#                                             nsweeps=nsweeps,
#                                             multi_phase=multi_phase,
#                                             num_phase=num_phase)
#     assert shape_updater.shape_move is vertex_move
#     assert np.allclose(shape_updater.move_ratio, move_ratio, rtol=1e-4)
#     assert shape_updater.trigger is trigger
#     assert shape_updater.nselect == nselect
#     assert shape_updater.nsweeps == nsweeps
#     assert shape_updater.multi_phase == multi_phase
#     assert shape_updater.num_phase == num_phase
#
#     move_ratio = 0.5
#     trigger = hoomd.trigger.Periodic(10)
#     nselect = 2
#     nsweeps = 4
#     multi_phase = False
#     num_phase = 1
#     shape_updater.move_ratio = move_ratio
#     shape_updater.trigger = trigger
#     shape_updater.nselect = nselect
#     shape_updater.nsweeps = nsweeps
#     shape_updater.multi_phase = multi_phase
#     shape_updater.num_phase = num_phase
#     assert np.allclose(shape_updater.move_ratio, move_ratio, rtol=1e-4)
#     assert shape_updater.trigger is trigger
#     assert shape_updater.nselect == nselect
#     assert shape_updater.nsweeps == nsweeps
#     assert shape_updater.multi_phase == multi_phase
#     assert shape_updater.num_phase == num_phase
#
#     assert shape_updater.shape_moves is None
#     assert shape_updater.shape_move_energy is None
#     assert shape_updater.particle_volume is None


verts = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
                    [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])


shape_move_classes = [VertexShapeMove, PythonShapeMove, ElasticShapeMove]

updater_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10),
         default_step_size=0.1),
    dict(trigger = hoomd.trigger.After(100),
         nselect = 2,
         nsweeps = 4,
         default_step_size = 0.5),
    dict(trigger = hoomd.trigger.Before(100),
         nselect = 2,
         nsweeps = 4,
         default_step_size = 0.5),
    dict(trigger=hoomd.trigger.Periodic(1000),
         nselect = 2,
         nsweeps = 4,
         default_step_size = 0.5)
]

python_move_constructor_args = [
    dict(
        callback = lambda typeid, param_list: {"vertices": verts.tolist(),
                                               "sweep_radius": 0,
                                               "ignore_statistics": True},
        param_move_probability = 1
        ),
    dict(
        callback = lambda typeid, param_list: {"vertices": (verts/2).tolist(),
                                               "sweep_radius": 0.05,
                                               "ignore_statistics": False},
        param_move_probability = 0.4
        ),
]

@pytest.mark.parametrize("shape_move_constructor_args", python_move_constructor_args)
def test_valid_construction_shape_moves(constructor_args):
    move = hpmc.shape_move.PythonShapeMove(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(move, attr) == value

@pytest.mark.parametrize("constructor_args", python_move_constructor_args)
@pytest.mark.parametrize("updater_constructor_args", updater_constructor_args)
@pytest.mark.parametrize("ShapeMoveClass", shape_move_classes)
def test_valid_construction_and_attach_python_shape_move(device,
        simulation_factory, two_particle_snapshot_factory, constructor_args):

        move = hpmc.shape_move.ShapeMoveClass(**python_move_constructor_args)

        updater = hpmc.update.Shape(**updater_constructor_args)
        updater.shape_move = move

        mc = hoomd.hpmc.integrate.ConvexPolyhedron()
        mc.shape['A'] = dict(vertices=verts)

        # create simulation & attach objects
        sim = simulation_factory(two_particle_snapshot_factory())
        sim.operations.integrator = mc
        sim.operations += updater

        # catch the error if running on the GPU
        if isinstance(device, hoomd.device.GPU):
            with pytest.raises(RuntimeError):
                sim.run(0)

         sim.run(0)

        assert move._attached
        assert updater._attached

         # validate the params were set properly
         for attr, value in python_move_constructor_args.items():
             assert getattr(move, attr) == value

         # validate the params were set properly
         for attr, value in updater_constructor_args.items():
             assert getattr(updater, attr) == value



@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_shape_move(attr, value):
    """Test that BoxMC can get and set attributes."""
    boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(10),
                                    betaP=10)

    setattr(boxmc, attr, value)
    if isinstance(value, dict):
        # check if we have the same keys
        assert value.keys() == getattr(boxmc, attr).keys()
        for k in value.keys():
            assert _is_close(value[k], getattr(boxmc, attr)[k])
    else:
        assert getattr(boxmc, attr) == value



def test_after_attaching(device, simulation_factory, lattice_snapshot_factory):
    verts = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
                        [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) / 2
    particle_volume = 1.0
    vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01},
                                               param_ratio=0.2,
                                               volume=particle_volume)
    move_ratio = 0.5
    trigger = hoomd.trigger.Periodic(10)
    nselect = 2
    nsweeps = 1
    multi_phase = True
    num_phase = 2
    shape_updater = hoomd.hpmc.update.Shape(shape_move=vertex_move,
                                            move_ratio=move_ratio,
                                            trigger=trigger,
                                            nselect=nselect,
                                            nsweeps=nsweeps,
                                            multi_phase=multi_phase,
                                            num_phase=num_phase)
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(0)
    mc.shape['A'] = {'vertices': verts}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=3))
    N = sim.state.snapshot.particles.N
    sim.seed = 0
    sim.operations.add(mc)
    sim.operations.add(shape_updater)
    sim.run(0)

    move_ratio = 1.0
    trigger = hoomd.trigger.Periodic(1)
    nselect = 1
    nsweeps = 4
    multi_phase = False
    num_phase = 1
    shape_updater.move_ratio = move_ratio
    shape_updater.trigger = trigger
    shape_updater.nselect = nselect
    shape_updater.nsweeps = nsweeps
    shape_updater.multi_phase = multi_phase
    shape_updater.num_phase = num_phase
    assert np.allclose(shape_updater.move_ratio, move_ratio, rtol=1e-4)
    assert shape_updater.trigger is trigger
    assert shape_updater.nselect == nselect
    assert shape_updater.nsweeps == nsweeps
    assert shape_updater.multi_phase == multi_phase
    assert shape_updater.num_phase == num_phase

    sim.run(2)
    assert sum(shape_updater.shape_moves) == 8
    assert shape_updater.shape_move_energy == 0.0
    assert shape_updater.particle_volume == N * particle_volume

    logging_check(
        hoomd.hpmc.update.Shape, ('hpmc', 'update'), {
            'shape_moves': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'shape_move_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'particle_volume': {
                'category': LoggerCategories.scalar,
                'default': True
            }
        })
