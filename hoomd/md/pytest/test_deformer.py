# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import hoomd.md as md
import numpy as np
import pytest


# Function for pbc wrapping
def pbc_wrap(x, y, z, Lx, Ly, Lz, xy, xz, yz):
    lo_x, hi_x = -Lx / 2, Lx / 2
    lo_y, hi_y = -Ly / 2, Ly / 2
    lo_z, hi_z = -Lz / 2, Lz / 2

    tilt_x = (xz - xy * yz) * z + xy * y
    if x >= hi_x + tilt_x:
        x -= Lx
    elif x < lo_x + tilt_x:
        x += Lx

    tilt_y = yz * z
    if y >= hi_y + tilt_y:
        y -= Ly
        x -= Ly * xy
    elif y < lo_y + tilt_y:
        y += Ly
        x += Ly * xy

    if z >= hi_z:
        z -= Lz
        y -= Lz * yz
        x -= Lz * xz
    elif z < lo_z:
        z += Lz
        y += Lz * yz
        x += Lz * xz

    return (x, y, z)


class TestLeesEdwardsBoxDeformer:
    # Test basic functionalities of the Lees-Edwards deformer
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
            (0.6, -0.1, 0.2),
        ],
    )
    @pytest.mark.parametrize("shear_rate", [0.01, -0.1, 1.0])
    @pytest.mark.parametrize(
        "max_xy",
        [0.3, 0.5, 1.0],  # default value is 0.5
    )
    def test_create(
        self, simulation_factory, box_lengths, initial_tilts, shear_rate, max_xy
    ):
        Lx, Ly, Lz = box_lengths
        xy0, xz0, yz0 = initial_tilts
        snap = hoomd.Snapshot()
        if snap.communicator.rank == 0:
            snap.particles.N = 3
            snap.particles.types = ["A"]
            snap.configuration.box = [Lx, Ly, Lz, xy0, xz0, yz0]
            snap.particles.position[:] = [[1.0, 2.0, 3.0], [0.1, -0.2, 0.3], [0, 0, 0]]
        sim = simulation_factory(snap)

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
        assert np.allclose(new_box.L_rate, (0.0, 0.0, 0.0))
        assert np.allclose(new_box.tilts_rate, (0.0, 0.0, 0.0))

        sim.run(0)

        assert lees_edwards.shear_rate == shear_rate
        assert lees_edwards.max_tilt == max_xy
        new_box = sim.state.box
        assert np.allclose(new_box.L, (Lx, Ly, Lz))
        assert np.allclose(new_box.tilts, (xy0, xz0, yz0))
        assert np.allclose(new_box.L_rate, (0.0, 0.0, 0.0))
        assert np.allclose(new_box.tilts_rate, (0.0, 0.0, 0.0))

        # Re-check after a run step is completed
        sim.run(1)

        xy = xy0 + shear_rate * dt
        xy -= (
            np.floor((xy + max_xy) / (2 * max_xy)) * 2 * max_xy
        )  # box can flip, so remap xy

        new_box = sim.state.box
        assert np.allclose(new_box.L, (Lx, Ly, Lz))
        assert np.allclose(new_box.tilts, (xy, xz0, yz0))
        assert np.allclose(new_box.L_rate, (0.0, 0.0, 0.0))
        assert np.allclose(new_box.tilts_rate, (shear_rate, 0.0, 0.0))

        # Also check that xy is accumulating properly after multiple runs
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
    def test_particles_unchanged_no_flip(self, simulation_factory, tilts):
        xy, xz, yz = tilts
        pos = [[0.1, -0.2, 0.3], [1.0, 2.0, 3.0], [-0.5, 3.0, 0.0]]
        snap = hoomd.Snapshot()
        if snap.communicator.rank == 0:
            snap.particles.N = 3
            snap.particles.types = ["A"]
            snap.configuration.box = [10, 10, 10, xy, xz, yz]
            snap.particles.position[:] = pos
        sim = simulation_factory(snap)

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
            for i in range(snap.particles.N):
                assert np.allclose(snap.particles.position[i], pos[i])

    # Test particle remapping when box flips
    @pytest.mark.parametrize(
        "xy0, shear_rate, flip, pos",
        [
            # positive flip
            (
                0.41,
                1.0,
                1,
                [[2.0, 0.0, 0.0], [3.0, 2.0, 1.0], [-3.5, -2.5, 0.0]],
            ),
            # negative flip
            (
                -0.41,
                -1.0,
                -1,
                [[2.0, 0.0, 0.0], [3.0, 2.0, 1.0], [-3.5, -2.5, 0.0]],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "box_lengths",
        [(10.0, 10.0, 10.0), (10.0, 8.0, 6.0)],
    )
    def test_particle_remap_on_flip(
        self, simulation_factory, xy0, shear_rate, flip, pos, box_lengths
    ):
        Lx, Ly, Lz = box_lengths
        snap = hoomd.Snapshot()
        if snap.communicator.rank == 0:
            snap.particles.N = 3
            snap.particles.types = ["A"]
            snap.configuration.box = [Lx, Ly, Lz, xy0, 0, 0]
            snap.particles.position[:] = pos
        sim = simulation_factory(snap)

        dt = 0.1
        deformer = md.deformer.LeesEdwardsBoxDeformer(shear_rate)
        integrator = md.Integrator(dt=dt, deformer=deformer)
        sim.operations.integrator = integrator
        sim.run(1)

        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            for i in range(snap.particles.N):
                x_shifted = pos[i][0] - flip * Ly
                new_pos = pbc_wrap(
                    x_shifted,
                    pos[i][1],
                    pos[i][2],
                    Lx,
                    Ly,
                    Lz,
                    xy=sim.state.box.xy,
                    xz=sim.state.box.xz,
                    yz=sim.state.box.yz,
                )
                assert np.allclose(snap.particles.position[i], new_pos)
