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

    @staticmethod
    def _box_geometry(box):
        """Return the triclinic-box origin and lattice vectors.
        The lattice vectors are stored as rows, such that
            position = origin + fractional @ lattice_vectors
        """
        Lx, Ly, Lz, xy, xz, yz = np.asarray(box, dtype=float)

        lattice_vectors = np.array(
            [
                [Lx, 0.0, 0.0],
                [xy * Ly, Ly, 0.0],
                [xz * Lz, yz * Lz, Lz],
            ],
            dtype=float,
        )
        lo = np.array([-Lx / 2.0, -Ly / 2.0, -Lz / 2.0], dtype=float)
        origin = np.array(
            [
                lo[0] + xy * lo[1] + xz * lo[2],
                lo[1] + yz * lo[2],
                lo[2],
            ],
            dtype=float,
        )
        return origin, lattice_vectors

    @classmethod
    def _fractional_coordinates(cls, positions, box):
        """Return fractional coordinates in the given box."""
        origin, lattice_vectors = cls._box_geometry(box)
        return (positions - origin) @ np.linalg.inv(lattice_vectors)

    @classmethod
    def _coordinates_from_fraction(cls, fractional, box):
        """Return Cartesian coordinates from fractional coordinates."""
        origin, lattice_vectors = cls._box_geometry(box)
        return origin + fractional @ lattice_vectors

    @classmethod
    def _unwrap(cls, positions, images, box):
        """Return unwrapped particle coordinates."""
        _, lattice_vectors = cls._box_geometry(box)
        return positions + images @ lattice_vectors

    @pytest.mark.parametrize(
        "Lx, Ly, max_xy",
        [
            (10.0, 10.0, 0.5),
            (12.0, 8.0, 0.75),
            (8.0, 12.0, 1.0 / 3.0),
        ],
    )
    @pytest.mark.parametrize("flip_sign", [1, -1])
    def test_box_flip_and_remap(
        self,
        simulation_factory,
        Lx,
        Ly,
        max_xy,
        flip_sign,
    ):
        # Begin beyond the tilt threshold to trigger a flip; no particle motion (dt=0)
        xy0 = flip_sign * (max_xy + 0.05)
        shear_rate = 0.01
        dt = 0.0

        initial_box = np.array([Lx, Ly, 10, xy0, 0.1, 0.3], dtype=float)

        # Define the initial positions through fractional coordinates so that
        # they lie inside the initial box for both signs of xy0
        fractional_initial = np.array(
            [
                [0.90, 0.90, 0.50],
                [0.10, 0.10, 0.50],
                [0.50, 0.50, 0.50],
            ],
            dtype=float,
        )
        positions = self._coordinates_from_fraction(fractional_initial, initial_box)
        images = np.array(
            [
                [2, 1, 0],
                [-1, 2, 1],
                [3, -2, 0],
            ],
            dtype=np.int32,
        )

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

        # Read the actual state before the flip (in case initialization modified state)
        snap_before = sim.state.get_snapshot()
        if snap_before.communicator.rank == 0:
            positions_before = np.asarray(snap_before.particles.position, dtype=float)
            images_before = np.asarray(snap_before.particles.image, dtype=np.int32)
            box_before = np.asarray(snap_before.configuration.box, dtype=float)
            unwrapped_before = self._unwrap(positions_before, images_before, box_before)

        sim.run(1)

        # Now, read the state after the flip run.
        snap_after = sim.state.get_snapshot()
        if snap_after.communicator.rank == 0:
            positions_after = np.asarray(snap_after.particles.position, dtype=float)
            images_after = np.asarray(snap_after.particles.image, dtype=np.int32)
            box_after = np.asarray(snap_after.configuration.box, dtype=float)

            xy_unflipped = xy0 + shear_rate * dt
            xy_expected = (
                xy_unflipped
                - np.floor((xy_unflipped + max_xy) / (2.0 * max_xy)) * 2.0 * max_xy
            )
            # Confirm that the resulting tilt has the expected value
            assert np.isclose(box_after[3], xy_expected)

            # Confirm that wrapped coordinates lie inside the flipped box
            fractional_after = self._fractional_coordinates(positions_after, box_after)
            assert np.all(fractional_after >= -1e-6)
            assert np.all(fractional_after < 1.0 + 1e-6)

            # Confirm that unwrapped coordinates after and before are the same
            unwrapped_after = self._unwrap(positions_after, images_after, box_after)
            assert np.allclose(unwrapped_after, unwrapped_before)
