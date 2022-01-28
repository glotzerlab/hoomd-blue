# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import hoomd.hpmc
import numpy as np
import pytest
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check


def test_before_attaching():
    verts = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
                        [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) / 2
    vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01},
                                               param_ratio=0.2,
                                               volume=1.0)
    move_ratio = 1.0
    trigger = hoomd.trigger.Periodic(1)
    nselect = 1
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
    assert shape_updater.shape_move is vertex_move
    assert np.allclose(shape_updater.move_ratio, move_ratio, rtol=1e-4)
    assert shape_updater.trigger is trigger
    assert shape_updater.nselect == nselect
    assert shape_updater.nsweeps == nsweeps
    assert shape_updater.multi_phase == multi_phase
    assert shape_updater.num_phase == num_phase

    move_ratio = 0.5
    trigger = hoomd.trigger.Periodic(10)
    nselect = 2
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

    assert shape_updater.shape_moves is None
    assert shape_updater.shape_move_energy is None
    assert shape_updater.particle_volume is None


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
