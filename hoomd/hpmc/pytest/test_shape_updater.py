import hoomd
import hoomd.hpmc
import numpy as np
import pytest


def test_before_attaching():
    verts = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
                        [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) / 2
   vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01}, param_ratio=0.2, volume=1.0)
    move_ratio = 1.0
    trigger = hoomd.trigger.Periodic(1)
    nselect = 1
    shape_updater = hoomd.hpmc.update.Shape(shape_move=vertex_move,
                                            move_ratio=move_ratio,
                                            trigger=trigger,
                                            nselect=nselect)
    assert shape_updater.shape_move is constant_move
    assert np.allclose(shape_updater.move_ratio, move_ratio, rtol=1e-4)
    assert shape_updater.trigger is trigger
    assert shape_updater.nselect == nselect

    move_ratio = 0.5
    trigger = hoomd.trigger.Periodic(10)
    nselect = 2
    shape_updater.move_ratio = move_ratio
    shape_updater.trigger = trigger
    shape_updater.nselect = nselect
    assert np.allclose(shape_updater.move_ratio, move_ratio, rtol=1e-4)
    assert shape_updater.trigger is trigger
    assert shape_updater.nselect == nselect

    assert shape_updater.shape_moves is None
    assert shape_updater.shape_move_energy is None


def test_after_attaching(device, simulation_factory, lattice_snapshot_factory):
    verts = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
                        [-1, 1, 1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]) / 2
    vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01}, param_ratio=0.2, volume=1.0)
    move_ratio = 0.5
    trigger = hoomd.trigger.Periodic(10)
    nselect = 2
    shape_updater = hoomd.hpmc.update.Shape(shape_move=vertex_move,
                                            move_ratio=move_ratio,
                                            trigger=trigger,
                                            nselect=nselect,
                                            nsweeps=1)

    mc = hoomd.hpmc.integrate.ConvexPolyhedron(0)
    mc.shape['A'] = {'vertices': verts}
    sim = simulation_factory(lattice_snapshot_factory(dimensions=3, a=2.0, n=3))
    sim.seed = 0
    sim.operations.add(mc)
    sim.operations.add(shape_updater)
    sim.run(0)

    move_ratio = 1.0
    trigger = hoomd.trigger.Periodic(1)
    nselect = 1
    shape_updater.move_ratio = move_ratio
    shape_updater.trigger = trigger
    shape_updater.nselect = nselect
    assert np.allclose(shape_updater.move_ratio, move_ratio, rtol=1e-4)
    assert shape_updater.trigger is trigger
    assert shape_updater.nselect == nselect

    sim.run(2)
    assert sum(shape_updater.shape_moves) == 2
    assert shape_updater.shape_move_energy == 0.0
