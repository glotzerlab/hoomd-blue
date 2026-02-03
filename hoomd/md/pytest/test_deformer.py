# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import hoomd.md as md
import numpy as np
import pytest


# Helper function
def create_simulation(box, positions, images=None):
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.particles.N = len(positions)
        snap.particles.position[:] = positions
        snap.particles.types = ["A"]
        if images is not None:
            snap.particles.image[:] = images
        snap.configuration.box = box

    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_snapshot(snap)
    return sim


# Test to check basic functionalities of the Lees-Edwards deformer
@pytest.mark.parametrize(
    "box_lengths",
    [
        (10.0, 10.0, 10.0),
        (30.0, 20.0, 10.0),
    ],
)
@pytest.mark.parametrize(
    "initial_tilts",
    [
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (-0.3, 0.2, 0.1),
    ],
)
@pytest.mark.parametrize("shear_rate", [0.0, 0.01, -0.1, 1.0])
@pytest.mark.parametrize(
    "max_xy",
    [0.3, 0.5, 1.0],  # default value is 0.5
)
def test_box_deformation_functionality(box_lengths, initial_tilts, shear_rate, max_xy):
    Lx, Ly, Lz = box_lengths
    xy0, xz0, yz0 = initial_tilts
    box = [Lx, Ly, Lz, xy0, xz0, yz0]
    sim = create_simulation(box, [[0.1, -0.2, 0.3]], [[0, 0, 0]])

    dt = 0.1

    lees_edwards = md.deformer.LeesEdwardsBoxDeformer(
        shear_rate=shear_rate, max_tilt=max_xy
    )
    integrator = md.Integrator(dt=dt, deformer=lees_edwards)
    sim.operations.integrator = integrator

    # Check for correctness of input arguments, box lengths, tilts, and deformations
    assert lees_edwards.shear_rate == shear_rate
    assert lees_edwards.max_tilt == max_xy
    new_box = sim.state.box
    assert np.allclose(new_box.L, (Lx, Ly, Lz))
    assert np.allclose(new_box.tilts, (xy0, xz0, yz0))
    # assert np.allclose(new_box.tilt_rates, (shear_rate, 0, 0))

    sim.run(0)

    assert lees_edwards.shear_rate == shear_rate
    assert lees_edwards.max_tilt == max_xy
    new_box = sim.state.box
    assert np.allclose(new_box.L, (Lx, Ly, Lz))
    assert np.allclose(new_box.tilts, (xy0, xz0, yz0))
    # assert np.allclose(new_box.tilt_rates, (shear_rate, 0, 0))

    # Re-check after a run step is completed
    sim.run(1)

    xy = xy0 + shear_rate * dt
    xy -= (
        np.floor((xy + max_xy) / (2 * max_xy)) * 2 * max_xy
    )  # box can flip, so remap xy

    new_box = sim.state.box
    assert np.allclose(new_box.L, (Lx, Ly, Lz))
    assert np.allclose(new_box.tilts, (xy, xz0, yz0))
    # assert np.allclose(new_box.tilt_rates, (shear_rate, 0, 0))

    # Also check that xy is accumulating properly after multiple runs, say T more steps
    xy_accumulated = new_box.xy
    T = 5
    sim.run(T)

    xy = xy_accumulated + T * shear_rate * dt
    xy -= np.floor((xy + max_xy) / (2 * max_xy)) * 2 * max_xy

    new_box = sim.state.box
    assert np.isclose(new_box.xy, xy)
    assert np.isclose(new_box.xz, xz0)
    assert np.isclose(new_box.yz, yz0)

    assert abs(sim.state.box.xy) <= lees_edwards.max_tilt


# Test that particles are not remapped when there is no flip
@pytest.mark.parametrize(
    "tilts",
    [
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (-0.2, 0.05, 0.0),
        (0.3, -0.1, 0.2),
        (0.49, 0.0, 0.0),
    ],
)
def test_particles_unchanged_no_flip(tilts):
    Lx, Ly, Lz = 10, 10, 10
    xy, xz, yz = tilts
    box = [Lx, Ly, Lz, xy, xz, yz]
    pos = [[1.0, 2.0, 3.0], [-0.5, 3.0, 0.0]]
    img = [[0, 0, 0], [1, 1, 1]]

    sim = create_simulation(box, pos, img)

    # With these values the box is not expected to flip in one step
    shear_rate = 0.1
    dt = 0.1
    max_tilt = 10.0

    le_deformer = md.deformer.LeesEdwardsBoxDeformer(shear_rate, max_tilt)
    integrator = md.Integrator(dt=dt, deformer=le_deformer)

    sim.operations.integrator = integrator
    sim.run(1)

    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        assert np.allclose(snap.particles.position[0], pos[0])
        assert np.all(snap.particles.image[0] == img[0])
        assert np.allclose(snap.particles.position[1], pos[1])
        assert np.all(snap.particles.image[1] == img[1])


# Test to check particle remapping when box flips
@pytest.mark.parametrize(
    "xy0, shear_rate, flip, pos, img",
    [
        # +1 flip
        (
            0.49,
            1.0,
            1,
            [[2.0, 0.0, 0.0], [5.0, 2.0, 1.0], [7.0, 5.0, 0.0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ),
        # +2 flips
        (
            0.45,
            6.0,
            2,
            [[1.0, 1.0, 1.0], [3.0, 2.0, 0.5], [4.0, 3.0, 0.0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ),
        # negative flip
        (
            -0.49,
            -1.0,
            -1,
            [[0.5, 0.0, 0.0], [1.5, 1.0, 0.0], [2.5, 2.0, 0.0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ),
    ],
)
def test_particle_remap_on_flip(xy0, shear_rate, flip, pos, img):
    Lx, Ly = 10.0, 10.0
    box = [Lx, Ly, 10, xy0, 0, 0]

    sim = create_simulation(box, pos, img)

    dt = 0.1

    deformer = md.deformer.LeesEdwardsBoxDeformer(shear_rate)
    integrator = md.Integrator(dt=dt, deformer=deformer)

    sim.operations.integrator = integrator
    sim.run(1)

    # New tilt after flip remains within limits
    assert abs(sim.state.box.xy) <= deformer.max_tilt
