# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test Image list generation for small box / large move size simulations."""

import hoomd
import pytest
import numpy


@pytest.fixture(scope="function")
def one_square_simulation(simulation_factory):
    """Construct a simulation box with one square.

    The box has L=1.2 and the square has side length 1. Some orientations of the
    square have no overlaps and some do.
    """
    snap = hoomd.Snapshot()
    snap.particles.N = 1
    snap.particles.types = ['A']
    snap.particles.position[:] = [[0, 0, 0]]
    snap.configuration.box = [1.2, 1.2, 0, 0, 0, 0]

    sim = simulation_factory(snap)
    mc = hoomd.hpmc.integrate.ConvexPolygon()
    mc.shape['A'] = dict(vertices=[
        (-0.5, -0.5),
        (0.5, -0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
    ])
    sim.operations.integrator = mc
    return sim


@pytest.mark.serial
@pytest.mark.cpu
def test_self_interaction_no_overlap(one_square_simulation):
    """Check that a properly aligned square does not overlap itself.

    When aligned to the box, it does not overlap with its own images.
    """
    sim = one_square_simulation
    mc = sim.operations.integrator

    with sim.state.cpu_local_snapshot as data:
        data.particles.orientation[0, :] = [1, 0, 0, 0]

    sim.operations._schedule()

    assert mc.overlaps == 0


@pytest.mark.serial
@pytest.mark.cpu
def test_self_interaction_overlap(one_square_simulation):
    """Check that a rotated square overlaps with itself.

    When aligned 45 degrees to the box, it overlaps with its own images.
    """
    sim = one_square_simulation
    mc = sim.operations.integrator

    with sim.state.cpu_local_snapshot as data:
        data.particles.orientation[0, :] = [
            0.9238795325112867, 0, 0, 0.3826834323650898
        ]

    sim.operations._schedule()

    assert mc.overlaps > 0


@pytest.mark.serial
@pytest.mark.cpu
@pytest.mark.validate
def test_self_interaction_run(one_square_simulation):
    """Check that one particle simulations operate correctly.

    This tests that particles are correctly checked against their own new
    orientation when checking i, j overlaps when i==j in a different image.
    """
    sim = one_square_simulation
    mc = sim.operations.integrator

    rotate_moves = numpy.zeros(2, dtype=numpy.uint64)
    translate_moves = numpy.zeros(2, dtype=numpy.uint64)

    for i in range(50000):
        sim.run(1)
        # ensure that no overlaps are present in any time step
        assert mc.overlaps == 0

        # tally the total number of moves
        rotate_moves += numpy.array(mc.rotate_moves, dtype=numpy.uint64)
        translate_moves += numpy.array(mc.translate_moves, dtype=numpy.uint64)

    # ensure that the simulation moved the particles
    assert rotate_moves[0] > 0
    assert rotate_moves[1] > 0
    assert translate_moves[0] > 0


@pytest.mark.serial
@pytest.mark.cpu
@pytest.mark.validate
def test_large_moves(simulation_factory, lattice_snapshot_factory):
    """Test that the nselect move size handling logic functions properly.

    Run a simulation with very large potential move distances in one timestep
    to ensure that the image list code considers all possible interactions.
    """
    snap = lattice_snapshot_factory(dimensions=2, a=2, n=16)

    sim = simulation_factory(snap)
    mc = hoomd.hpmc.integrate.ConvexPolygon(translation_move_probability=1.0,
                                            nselect=4,
                                            default_d=100)
    mc.shape['A'] = dict(vertices=[
        (-0.5, -0.5),
        (0.5, -0.5),
        (0.5, 0.5),
        (-0.5, 0.5),
    ])
    sim.operations.integrator = mc

    translate_moves = numpy.zeros(2, dtype=numpy.uint64)

    for i in range(50000):
        sim.run(1)
        # ensure that no overlaps are present in any time step
        assert mc.overlaps == 0

        # tally the total number of moves
        translate_moves += numpy.array(mc.translate_moves, dtype=numpy.uint64)

    # ensure that the simulation moved the particles
    assert translate_moves[0] > 0
    assert translate_moves[1] > 0
