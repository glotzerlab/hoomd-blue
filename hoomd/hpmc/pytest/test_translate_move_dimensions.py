# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.translate_move_dimensions."""

import hoomd
import pytest
import numpy as np

# octahedron vertices
vertices = [
    (-0.5, 0, 0),
    (0.5, 0, 0),
    (0, -0.5, 0),
    (0, 0.5, 0),
    (0, 0, -0.5),
    (0, 0, 0.5),
]


@pytest.fixture
def sim(device):
    """A small 3D simulation with 4 octahedra on a 2D grid."""
    simulation = hoomd.Simulation(device=device, seed=1)

    snap = hoomd.Snapshot(device.communicator)
    if device.communicator.rank == 0:
        snap.configuration.box = [6, 6, 6, 0, 0, 0]
        snap.particles.N = 4
        snap.particles.types = ["A"]
        snap.particles.typeid[:] = [0, 0, 0, 0]
        snap.particles.position[:] = [
            [-1.5, -1.5, 0],
            [1.5, -1.5, 0],
            [-1.5, 1.5, 0],
            [1.5, 1.5, 0],
        ]

    simulation.create_state_from_snapshot(snap)
    return simulation


def make_mc(sim, d=0.1, a=0.3):
    mc = hoomd.hpmc.integrate.ConvexPolyhedron()
    mc.shape["A"] = dict(vertices=vertices)
    mc.d["A"] = d
    mc.a["A"] = a
    sim.operations.integrator = mc
    sim.run(0)  # attach
    return mc


def test_default_is_none(sim):
    """translate_move_dimensions defaults to None (resolved on attach/use)."""
    mc = make_mc(sim)
    assert mc.translate_move_dimensions is None


@pytest.mark.parametrize("dims", [2, 3])
def test_set_and_get(sim, dims):
    mc = make_mc(sim)
    mc.translate_move_dimensions = dims
    assert mc.translate_move_dimensions == dims


@pytest.mark.parametrize("bad", [0, 1, 4, -1])
def test_invalid_value_raises(sim, bad):
    mc = make_mc(sim)
    with pytest.raises(RuntimeError):
        mc.translate_move_dimensions = bad


def test_z_positions_unchanged(sim):
    """With translate_move_dimensions=2, z coords must never change."""
    mc = make_mc(sim, d=0.2, a=0.0)
    mc.translation_move_probability = 1.0
    mc.translate_move_dimensions = 2

    snap_before = sim.state.get_snapshot()

    sim.run(10)

    snap_after = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        z_before = np.array(snap_before.particles.position[:, 2])
        z_after = np.array(snap_after.particles.position[:, 2])
        np.testing.assert_array_equal(
            z_before,
            z_after,
            err_msg="z changed despite translate_move_dimensions=2",
        )


def test_xy_positions_do_change(sim):
    """With dims=2, particles should still move in xy."""
    mc = make_mc(sim, d=0.2, a=0.0)
    mc.translation_move_probability = 1.0
    mc.translate_move_dimensions = 2

    snap_before = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        xy_before = np.array(snap_before.particles.position[:, :2])
    sim.run(10)

    snap_after = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        xy_after = np.array(snap_after.particles.position[:, :2])
        assert not np.allclose(xy_before, xy_after), (
            "xy positions never changed. translation moves are broken."
        )


def test_rotations_active_with_dims_2(sim):
    """Rotation moves should still be accepted when translate_move_dimensions=2."""
    mc = make_mc(sim, d=0.0, a=0.4)  # disable translation to isolate
    mc.translation_move_probability = 0.0
    mc.translate_move_dimensions = 2

    sim.run(10)

    r_acc, _ = mc.rotate_moves
    if sim.device.communicator.rank == 0:
        assert r_acc > 0, "Rotations broken by translate_move_dimensions=2."


def test_dims_3_allows_z_translation(sim):
    """With dims=3, z positions should eventually change."""
    mc = make_mc(sim, d=0.5, a=0.0)
    mc.translation_move_probability = 1.0
    mc.translate_move_dimensions = 3

    snap_before = sim.state.get_snapshot()

    sim.run(10)

    snap_after = sim.state.get_snapshot()
    if sim.device.communicator.rank == 0:
        z_before = np.array(snap_before.particles.position[:, 2])
        z_after = np.array(snap_after.particles.position[:, 2])
        assert not np.allclose(z_before, z_after), (
            "z positions never changed with dims=3."
        )
