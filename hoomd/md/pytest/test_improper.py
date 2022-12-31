# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import expected_loggable_params
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)
import pytest
import numpy

import itertools
# Test parameters include the class, improper params, force, and energy.
# This is parameterized to plan for any future expansion with additional
# improper potentials.
improper_test_parameters = [
    (
        hoomd.md.improper.Harmonic,
        dict(k=1.5, chi0=0.2),
        [
            [0.0, 0.0, -0.15049702126325717],
            [0.0, 0.0, 0.15049702126325717],
            [-0.014900695174580036, 0.0, -0.14900695174579917],
            [0.014900695174580036, 0.0, 0.14900695174579917],
        ],
        0.007549784469704433,
    ),
]


@pytest.mark.parametrize("improper_cls, params, force, energy",
                         improper_test_parameters)
def test_before_attaching(improper_cls, params, force, energy):
    potential = improper_cls()
    potential.params['A-A-A-A'] = params
    for key in params:
        assert potential.params['A-A-A-A'][key] == pytest.approx(params[key])


@pytest.fixture(scope='session')
def snapshot_factory(device):

    def make_snapshot():
        snapshot = hoomd.Snapshot(device.communicator)
        N = 4
        L = 10
        if snapshot.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            snapshot.configuration.box = box
            snapshot.particles.N = N
            snapshot.particles.types = ['A']
            # shift particle positions slightly in z so MPI tests pass
            snapshot.particles.position[:] = [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0.1],
            ]

            snapshot.impropers.N = 1
            snapshot.impropers.types = ['A-A-A-A']
            snapshot.impropers.typeid[0] = 0
            snapshot.impropers.group[0] = (0, 1, 2, 3)

        return snapshot

    return make_snapshot


@pytest.mark.parametrize("improper_cls, params, force, energy",
                         improper_test_parameters)
def test_after_attaching(snapshot_factory, simulation_factory, improper_cls,
                         params, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = improper_cls()
    potential.params['A-A-A-A'] = params

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(potential)

    langevin = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All())
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in params:
        assert potential.params['A-A-A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize("improper_cls, params, force, energy",
                         improper_test_parameters)
def test_forces_and_energies(snapshot_factory, simulation_factory, improper_cls,
                             params, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = improper_cls()
    potential.params['A-A-A-A'] = params

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(potential)

    langevin = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All())
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    sim_energies = potential.energies
    sim_forces = potential.forces

    if sim.device.communicator.rank == 0:
        assert sum(sim_energies) == pytest.approx(energy, rel=1e-4)
        numpy.testing.assert_allclose(sim_forces, force, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("improper_cls, params, force, energy",
                         improper_test_parameters)
def test_kernel_parameters(snapshot_factory, simulation_factory, improper_cls,
                           params, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = improper_cls()
    potential.params['A-A-A-A'] = params

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(potential)

    langevin = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All())
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    autotuned_kernel_parameter_check(instance=potential,
                                     activate=lambda: sim.run(1))


# Test Logging
@pytest.mark.parametrize(
    'cls, expected_namespace, expected_loggables',
    zip((hoomd.md.improper.Improper, hoomd.md.improper.Harmonic),
        itertools.repeat(('md', 'improper')),
        itertools.repeat(expected_loggable_params)))
def test_logging(cls, expected_namespace, expected_loggables):
    logging_check(cls, expected_namespace, expected_loggables)


# Test pickling
@pytest.mark.parametrize("improper_cls, params, force, energy",
                         improper_test_parameters)
def test_pickling(simulation_factory, snapshot_factory, improper_cls, params,
                  force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = improper_cls()
    potential.params['A-A-A-A'] = params

    pickling_check(potential)

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(potential)

    langevin = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All())
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    pickling_check(potential)
