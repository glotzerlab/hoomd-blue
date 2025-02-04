# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest


@pytest.fixture(scope='session')
def snapshot_factory(device):
    """Make a test snapshot for update_group_dof testing."""

    def make_snapshot():
        """Make the snapshot."""
        snap = hoomd.Snapshot(device.communicator)

        if snap.communicator.rank == 0:
            box = [10, 10, 10, 0, 0, 0]
            snap.configuration.box = box
            snap.particles.N = 3
            snap.particles.position[:] = [[0, 1, 0], [-1, 1, 0], [1, 1, 0]]
            snap.particles.velocity[:] = [[0, 0, 0], [0, -1, 0], [0, 1, 0]]
            snap.particles.moment_inertia[:] = [
                [2.0, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
            ]
            snap.particles.angmom[:] = [[0, 2, 4, 6]] * 3
            snap.particles.types = ['A', 'B']

        return snap

    return make_snapshot


def test_set_snapshot(simulation_factory, snapshot_factory):
    """Test that the number of degrees of freedom updates after set_snapshot."""
    snapshot = snapshot_factory()

    sim = simulation_factory(snapshot)
    filter_all = hoomd.filter.All()
    method = hoomd.md.methods.Langevin(filter=filter_all, kT=1.0)
    sim.operations.integrator = hoomd.md.Integrator(
        0.005,
        methods=[method],
        integrate_rotational_dof=True,
    )
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filter_all)
    sim.operations.add(thermo)

    # check initial degrees of freedom in snapshot
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 9
    assert thermo.rotational_degrees_of_freedom == 7

    # add a particle and check after set_snapshot
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = 4
        snapshot.particles.moment_inertia[3] = [0, 0, 1]
    sim.state.set_snapshot(snapshot)

    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 12
    assert thermo.rotational_degrees_of_freedom == 8


def test_local_snapshot(simulation_factory, snapshot_factory):
    """Test dof doesn't update after local snapshot release."""
    snapshot = snapshot_factory()

    sim = simulation_factory(snapshot)
    filter_all = hoomd.filter.All()
    method = hoomd.md.methods.Langevin(filter=filter_all, kT=1.0)
    sim.operations.integrator = hoomd.md.Integrator(
        0.005,
        methods=[method],
        integrate_rotational_dof=True,
    )
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filter_all)
    sim.operations.add(thermo)

    sim.run(0)
    assert thermo.rotational_degrees_of_freedom == 7

    # reduce the rotational dof by modifying the moment of inertia of particle 0
    with sim.state.cpu_local_snapshot as snapshot:
        N = len(snapshot.particles.position)
        idx = snapshot.particles.rtag[0]

        if idx < N:
            snapshot.particles.moment_inertia[idx] = [0, 0, 0]

    # group dof doesn't automatically update with local snapshots
    sim.run(0)
    assert thermo.rotational_degrees_of_freedom == 7

    # test the update after manually calling
    sim.state.update_group_dof()
    sim.run(0)
    assert thermo.rotational_degrees_of_freedom == 6


def test_set_integrator(simulation_factory, snapshot_factory):
    """Test dof update after setting an integrator."""
    snapshot = snapshot_factory()

    sim = simulation_factory(snapshot)
    filter_all = hoomd.filter.All()
    method1 = hoomd.md.methods.Langevin(filter=filter_all, kT=1.0)
    integrator1 = hoomd.md.Integrator(
        0.005,
        methods=[method1],
        integrate_rotational_dof=True,
    )
    thermostat = hoomd.md.methods.thermostats.Bussi(kT=1.0)
    method2 = hoomd.md.methods.ConstantVolume(filter=filter_all,
                                              thermostat=thermostat)
    integrator2 = hoomd.md.Integrator(
        0.005,
        methods=[method2],
        integrate_rotational_dof=True,
    )

    sim.operations.integrator = integrator1
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filter_all)
    sim.operations.add(thermo)

    # check degrees of freedom for Langevin
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 9
    assert thermo.rotational_degrees_of_freedom == 7

    # check the degrees of freedom for NVT
    sim.operations.integrator = integrator2
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 6
    assert thermo.rotational_degrees_of_freedom == 7


def test_set_method(simulation_factory, snapshot_factory):
    """Test dof update after setting an integration method."""
    snapshot = snapshot_factory()

    sim = simulation_factory(snapshot)
    filter_all = hoomd.filter.All()
    method1 = hoomd.md.methods.Langevin(filter=filter_all, kT=1.0)
    integrator = hoomd.md.Integrator(
        0.005,
        methods=[method1],
        integrate_rotational_dof=True,
    )
    thermostat = hoomd.md.methods.thermostats.Bussi(kT=1.0)
    method2 = hoomd.md.methods.ConstantVolume(filter=filter_all,
                                              thermostat=thermostat)

    sim.operations.integrator = integrator
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filter_all)
    sim.operations.add(thermo)

    # check degrees of freedom for Langevin
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 9
    assert thermo.rotational_degrees_of_freedom == 7

    # check the degrees of freedom for NVT
    sim.operations.integrator.methods[0] = method2
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 6
    assert thermo.rotational_degrees_of_freedom == 7

    # check after deleting a method
    sim.operations.integrator.methods.pop()
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 0
    assert thermo.rotational_degrees_of_freedom == 0


def test_set_integrate_rotational_dof(simulation_factory, snapshot_factory):
    """Test dof update after setting integrate_rotational_dof."""
    snapshot = snapshot_factory()

    sim = simulation_factory(snapshot)
    filter_all = hoomd.filter.All()
    method = hoomd.md.methods.Langevin(filter=filter_all, kT=1.0)
    integrator = hoomd.md.Integrator(
        0.005,
        methods=[method],
        integrate_rotational_dof=True,
    )

    sim.operations.integrator = integrator
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filter_all)
    sim.operations.add(thermo)

    # check degrees of freedom with setting True
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 9
    assert thermo.rotational_degrees_of_freedom == 7

    # check the degrees of freedom setting False
    sim.operations.integrator.integrate_rotational_dof = False
    sim.run(0)
    assert thermo.rotational_degrees_of_freedom == 0


def test_filter_updater(simulation_factory, snapshot_factory):
    """Test dof update after filter updater triggers."""
    snapshot = snapshot_factory()

    sim = simulation_factory(snapshot)
    filter_type = hoomd.filter.Type(['A'])
    method = hoomd.md.methods.Langevin(filter=filter_type, kT=1.0)
    integrator = hoomd.md.Integrator(
        0.005,
        methods=[method],
        integrate_rotational_dof=True,
    )

    sim.operations.integrator = integrator
    filter_all = hoomd.filter.All()
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=filter_all)
    sim.operations.add(thermo)

    # check initial degrees of freedom
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 9
    assert thermo.rotational_degrees_of_freedom == 7

    with sim.state.cpu_local_snapshot as snapshot:
        snapshot.particles.typeid[:] = 1

    # group hasn't updated, DOF should remain the same
    sim.run(0)
    assert thermo.translational_degrees_of_freedom == 9
    assert thermo.rotational_degrees_of_freedom == 7

    # add the filter updater and trigger it to change the DOF
    filter_updater = hoomd.update.FilterUpdater(
        trigger=hoomd.trigger.Periodic(1), filters=[filter_type])
    sim.operations.updaters.append(filter_updater)

    sim.run(1)
    assert thermo.translational_degrees_of_freedom == 0
    assert thermo.rotational_degrees_of_freedom == 0
