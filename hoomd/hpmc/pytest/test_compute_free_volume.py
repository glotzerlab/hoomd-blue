# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import math
import pytest
import numpy as np
from hoomd.error import DataAccessError
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check, autotuned_kernel_parameter_check


def test_before_attaching():
    free_volume = hoomd.hpmc.compute.FreeVolume(test_particle_type="B", num_samples=100)

    assert free_volume.test_particle_type == "B"
    assert free_volume.num_samples == 100
    with pytest.raises(DataAccessError):
        free_volume.free_volume


def test_after_attaching(simulation_factory, lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=["A", "B"])
    sim = simulation_factory(snap)
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape["A"] = {"diameter": 1.0}
    mc.shape["B"] = {"diameter": 0.2}
    sim.operations.add(mc)

    free_volume = hoomd.hpmc.compute.FreeVolume(test_particle_type="B", num_samples=100)

    sim.operations.add(free_volume)
    assert len(sim.operations.computes) == 1
    sim.run(0)

    assert free_volume.test_particle_type == "B"
    assert free_volume.num_samples == 100

    sim.run(10)
    assert isinstance(free_volume.free_volume, float)


_radii = [
    (0.25, 0.05),
    (0.4, 0.05),
    (0.7, 0.17),
]


@pytest.mark.parametrize("radius1, radius2", _radii)
def test_validation_systems(
    simulation_factory, lattice_snapshot_factory, radius1, radius2
):
    n = 7
    free_volume = (n**3) * (1 - (4 / 3) * np.pi * (radius1 + radius2) ** 3)
    free_volume = max([0.0, free_volume])
    sim = simulation_factory(
        lattice_snapshot_factory(particle_types=["A", "B"], n=n, a=1, dimensions=3, r=0)
    )

    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape["A"] = {"diameter": radius1 * 2}
    mc.shape["B"] = {"diameter": radius2 * 2}
    sim.operations.add(mc)

    free_volume_compute = hoomd.hpmc.compute.FreeVolume(
        test_particle_type="B", num_samples=10000
    )
    sim.operations.add(free_volume_compute)
    sim.run(0)
    # rtol is fairly high as the free volume available to a sized particle
    # is less than the total available volume
    np.testing.assert_allclose(free_volume, free_volume_compute.free_volume, rtol=2e-2)

    # Tet the kernel parameter tuner.
    def activate_tuner():
        sim.run(1)
        # We need to make the kernel be called.
        free_volume_compute.free_volume

    autotuned_kernel_parameter_check(
        instance=free_volume_compute, activate=activate_tuner
    )


def test_logging():
    logging_check(
        hoomd.hpmc.compute.FreeVolume,
        ("hpmc", "compute"),
        {"free_volume": {"category": LoggerCategories.scalar, "default": True}},
    )


def test_2d_free_volume(simulation_factory):
    snapshot = hoomd.Snapshot()
    if snapshot.communicator.rank == 0:
        snapshot.configuration.box = (100, 100, 0, 0, 0, 0)
        snapshot.particles.N = 1
        snapshot.particles.types = ["A"]

    sim = simulation_factory(snapshot)

    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape["A"] = dict(diameter=1)

    free_volume = hoomd.hpmc.compute.FreeVolume(
        test_particle_type="A", num_samples=100000
    )

    sim.operations.integrator = mc
    sim.operations.computes.append(free_volume)

    sim.run(0)
    f = free_volume.free_volume
    if snapshot.communicator.rank == 0:
        assert f == pytest.approx(expected=100 * 100 - math.pi, rel=1e-3)


def test_free_volume_export_name_matches_integrator(
    device, simulation_factory, two_particle_snapshot_factory, valid_args
):
    integrator = valid_args[0]
    args = valid_args[1]
    n_dimensions = valid_args[2]

    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()

    if integrator == hoomd.hpmc.integrate.Sphinx and isinstance(
        device, hoomd.device.GPU
    ):
        pytest.skip("Sphinx does not build on the GPU by default.")

    mc = integrator()
    mc.shape["A"] = args
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=n_dimensions))
    assert sim.operations.integrator is None
    sim.operations.add(mc)
    sim.operations._schedule()

    free_vol = hoomd.hpmc.compute.FreeVolume(test_particle_type="A", num_samples=10)
    sim.operations.computes.append(free_vol)

    try:
        sim.run(1)
    except AttributeError as e:
        pytest.fail(
            f"FreeVolume export class name mismatch for {integrator.__name__}: {e}"
        )
