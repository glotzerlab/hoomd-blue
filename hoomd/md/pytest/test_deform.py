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
        lees_edwards = md.deform.LeesEdwardsBoxDeformer(
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
        xy -= np.floor((xy + max_xy) / (2.0 * max_xy)) * 2.0 * max_xy

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

    @pytest.mark.parametrize("max_xy", [0.3, 0.5, 0.8])
    @pytest.mark.parametrize(
        "flip_case, shear_rate",
        [
            ("none", 0.9),
            ("positive", 0.9),
            ("negative", -0.9),
        ],
    )
    def test_box_flip_and_remap(
        self,
        simulation_factory,
        max_xy,
        flip_case,
        shear_rate,
    ):
        Lx = Ly = Lz = 10.0
        dt = 0.1

        # Start from positions unaffected by base class wrapping
        if flip_case == "none":
            xy0 = 0.5 * max_xy
            positions = np.array(
                [
                    [2.0, 2.0, 0.0],
                    [-2.0, -2.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0],
                ],
            )
        elif flip_case == "positive":
            xy0 = max_xy - 0.05
            positions = np.array(
                [
                    [4.8, 4.0, 0.0],
                    [-4.8, -4.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0],
                ],
                dtype=float,
            )
        elif flip_case == "negative":
            xy0 = -max_xy + 0.05
            positions = np.array(
                [
                    [-4.8, 4.0, 0.0],
                    [4.8, -4.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, -1.0, 1.0],
                ],
                dtype=float,
            )
        images = np.array(
            [
                [2, 1, 0],
                [-1, 2, 1],
                [3, -2, 0],
                [-2, -1, -1],
            ],
            dtype=np.int32,
        )
        initial_box = [Lx, Ly, Lz, xy0, 0.1, 0.3]

        snap = hoomd.Snapshot()
        if snap.communicator.rank == 0:
            snap.particles.N = len(positions)
            snap.particles.types = ["A"]
            snap.configuration.box = initial_box
            snap.particles.position[:] = positions
            snap.particles.image[:] = images
        sim = simulation_factory(snap)
        deformer = md.deform.LeesEdwardsBoxDeformer(
            shear_rate=shear_rate, max_tilt=max_xy
        )
        sim.operations.integrator = md.Integrator(dt=dt, deformer=deformer)

        sim.run(1)

        snap_after = sim.state.get_snapshot()
        if snap_after.communicator.rank == 0:
            xy_unflipped = xy0 + shear_rate * dt
            xy_flipped = xy_unflipped - (
                np.floor((xy_unflipped + max_xy) / (2.0 * max_xy)) * 2.0 * max_xy
            )

            # Image update from Lees-Edwards flip
            shift_value = (xy_flipped - xy_unflipped) * Ly / Lx
            if shift_value >= 0:
                image_shift = int(np.floor(shift_value + 0.5))
            else:
                image_shift = int(np.ceil(shift_value - 0.5))

            expected_images = images.copy()
            expected_images[:, 0] -= images[:, 1] * image_shift

            # Wrap positions into the flipped box
            expected_positions = positions.copy()
            Lx, Ly, Lz, xy, xz, yz = snap_after.configuration.box
            lo = np.array([-Lx / 2.0, -Ly / 2.0, -Lz / 2.0])

            # Compute fractional coordinates
            delta = expected_positions - lo
            delta[:, 0] -= (xz - yz * xy) * expected_positions[
                :, 2
            ] + xy * expected_positions[:, 1]
            delta[:, 1] -= yz * expected_positions[:, 2]
            fpos = delta / np.array([Lx, Ly, Lz])

            # Integer box crossings required to wrap particles into the primary box.
            wrap_shift = np.floor(fpos).astype(np.int32)
            fpos -= wrap_shift
            expected_images += wrap_shift

            # All final fractional coordinates must lie inside [0, 1).
            assert np.all((fpos >= -1e-6) & (fpos < 1.0 + 1e-6))

            # Convert fractional coordinates back to Cartesian coordinates
            expected_positions = lo + fpos * np.array([Lx, Ly, Lz])
            expected_positions[:, 0] += (
                xy * expected_positions[:, 1] + xz * expected_positions[:, 2]
            )
            expected_positions[:, 1] += yz * expected_positions[:, 2]

            # Assertions on expected tilt, images, and positions
            assert np.isclose(snap_after.configuration.box[3], xy_flipped)
            assert np.array_equal(snap_after.particles.image, expected_images)
            assert np.allclose(snap_after.particles.position, expected_positions)
