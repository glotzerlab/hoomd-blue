# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import hoomd.md as md
import numpy as np
import pytest


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
        assert np.allclose(new_box.L_rates, (0.0, 0.0, 0.0))
        assert np.allclose(new_box.tilt_rates, (0.0, 0.0, 0.0))

        sim.run(0)

        assert lees_edwards.shear_rate == shear_rate
        assert lees_edwards.max_tilt == max_xy
        new_box = sim.state.box
        assert np.allclose(new_box.L, (Lx, Ly, Lz))
        assert np.allclose(new_box.tilts, (xy0, xz0, yz0))
        assert np.allclose(new_box.L_rates, (0.0, 0.0, 0.0))
        assert np.allclose(new_box.tilt_rates, (0.0, 0.0, 0.0))

        # Re-check after a run step is completed
        sim.run(1)

        xy = xy0 + shear_rate * dt
        xy -= (
            np.floor((xy + max_xy) / (2 * max_xy)) * 2 * max_xy
        )  # box can flip, so remap xy

        new_box = sim.state.box
        assert np.allclose(new_box.L, (Lx, Ly, Lz))
        assert np.allclose(new_box.tilts, (xy, xz0, yz0))
        assert np.allclose(new_box.L_rates, (0.0, 0.0, 0.0))
        assert np.allclose(new_box.tilt_rates, (shear_rate, 0.0, 0.0))

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
            assert np.allclose(snap.particles.position[:], pos[:])

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
        pos = np.asarray(pos)
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

        xy, xz, yz = sim.state.box.xy, sim.state.box.xz, sim.state.box.yz
        x = pos[:, 0] - flip * Ly

        # Compute tilt correction and apply wrap
        tilt_x = (xz - xy * yz) * pos[:, 2] + xy * pos[:, 1]
        x_wrapped = np.where(
            x >= (Lx / 2) + tilt_x, x - Lx, np.where(x < (-Lx / 2) + tilt_x, x + Lx, x)
        )
        wrapped_pos = np.column_stack((x_wrapped, pos[:, 1], pos[:, 2]))

        if snap.communicator.rank == 0:
            assert np.allclose(snap.particles.position[:], wrapped_pos[:])
