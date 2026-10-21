# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd.conftest import pickling_check


def test_attributes():
    active = hoomd.md.force.CollectiveActive(filter=hoomd.filter.All())

    assert active.filter == hoomd.filter.All()
    assert active.active_force["A"] == (1.0, 0.0, 0.0)

    active.active_force["A"] = (0.5, 0.0, 0.0)
    assert active.active_force["A"] == (0.5, 0.0, 0.0)
    active.active_force["A"] = (0.0, 0.0, 1.0)
    assert active.active_force["A"] == (0.0, 0.0, 1.0)

    # no torque is applied
    with pytest.raises(AttributeError):
        active.active_torque


def test_attach(simulation_factory, two_particle_snapshot_factory):
    active = hoomd.md.force.CollectiveActive(filter=hoomd.filter.All())

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(0.05)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    assert active.active_force["A"] == (1.0, 0.0, 0.0)

    active.active_force["A"] = (0.5, 0.0, 0.0)
    assert active.active_force["A"] == (0.5, 0.0, 0.0)
    active.active_force["A"] = (0.0, 0.0, 1.0)
    assert active.active_force["A"] == (0.0, 0.0, 1.0)


@pytest.mark.parametrize("typeids", [(0, 0), (0, 1), (1, 1)])
def test_forces_shared_by_type(simulation_factory, two_particle_snapshot_factory, typeids):
    active = hoomd.md.force.CollectiveActive(filter=hoomd.filter.All())
    active.active_force["A"] = (1.0, 2.0, 3.0)
    active.active_force["B"] = (0.0, -1.0, 0.5)

    snapshot = two_particle_snapshot_factory(
        particle_types=["A", "B"], dimensions=3, d=8
    )
    if snapshot.communicator.rank == 0:
        snapshot.particles.typeid[:] = typeids
        # the orientations differ between the particles but must not matter
        snapshot.particles.orientation[:] = [
            [1.0, 0.0, 0.0, 0.0],
            [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        ]
    sim = simulation_factory(snapshot)
    integrator = hoomd.md.Integrator(0.05)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    snap = sim.state.get_snapshot()
    forces = active.forces
    torques = active.torques
    if snap.communicator.rank == 0:
        expected = {0: (1.0, 2.0, 3.0), 1: (0.0, -1.0, 0.5)}
        for f, t, typeid in zip(forces, torques, snap.particles.typeid):
            np.testing.assert_allclose(f, expected[typeid], atol=1e-6)
            np.testing.assert_allclose(t, 0.0, atol=1e-6)


def test_filter(simulation_factory, two_particle_snapshot_factory):
    active = hoomd.md.force.CollectiveActive(filter=hoomd.filter.Type(["A"]))

    snapshot = two_particle_snapshot_factory(
        particle_types=["A", "B"], dimensions=3, d=8
    )
    if snapshot.communicator.rank == 0:
        snapshot.particles.typeid[:] = (0, 1)
    sim = simulation_factory(snapshot)
    integrator = hoomd.md.Integrator(0.05)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    snap = sim.state.get_snapshot()
    forces = active.forces
    if snap.communicator.rank == 0:
        for f, typeid in zip(forces, snap.particles.typeid):
            if typeid == 0:
                np.testing.assert_allclose(f, (1.0, 0.0, 0.0), atol=1e-6)
            else:
                np.testing.assert_allclose(f, 0.0, atol=1e-6)


def test_no_diffusion_without_updater(simulation_factory, two_particle_snapshot_factory):
    active = hoomd.md.force.CollectiveActive(filter=hoomd.filter.All())
    active.active_force["A"] = (1.0, 2.0, 3.0)

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(0.05)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(20)

    np.testing.assert_allclose(active.active_force["A"], (1.0, 2.0, 3.0), atol=1e-6)


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    active = hoomd.md.force.CollectiveActive(filter=hoomd.filter.All())
    pickling_check(active)
    integrator = hoomd.md.Integrator(
        0.05,
        methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0)],
        forces=[active],
    )
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(active)
