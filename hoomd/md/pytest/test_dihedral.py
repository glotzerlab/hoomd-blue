# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd import md
from hoomd.conftest import expected_loggable_params
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)
import pytest
import numpy

import itertools
# Test parameters include the class, class keyword arguments, bond params,
# force, and energy.
dihedral_test_parameters = [
    (
        hoomd.md.dihedral.Periodic,
        dict(),
        dict(k=3.0, d=-1, n=2, phi0=numpy.pi / 2),
        0,
        3,
    ),
    (
        hoomd.md.dihedral.Periodic,
        dict(),
        dict(k=10.0, d=1, n=1, phi0=numpy.pi / 4),
        5.0,
        5.0,
    ),
    (
        hoomd.md.dihedral.Periodic,
        dict(),
        dict(k=5.0, d=1, n=3, phi0=numpy.pi / 6),
        1.9411,
        0.0852,
    ),
    (
        hoomd.md.dihedral.OPLS,
        dict(),
        dict(k1=1.0, k2=1.5, k3=0.5, k4=0.75),
        -0.616117,
        2.42678,
    ),
    (
        hoomd.md.dihedral.OPLS,
        dict(),
        dict(k1=0.5, k2=2.5, k3=1.5, k4=1.0),
        -0.732233,
        2.89645,
    ),
    (
        hoomd.md.dihedral.OPLS,
        dict(),
        dict(k1=2.0, k2=1.0, k3=0.25, k4=3.5),
        -0.0277282,
        5.74372,
    ),
    (
        hoomd.md.dihedral.Table,
        dict(width=2),
        dict(U=[0, 10], tau=[0, 1]),
        -0.375,
        3.75,
    ),
]


@pytest.fixture(scope='session')
def dihedral_snapshot_factory(device):

    def make_snapshot(d=1.0, phi_deg=45, particle_types=['A'], L=20):
        phi_rad = phi_deg * (numpy.pi / 180)
        # the central particles are along the x-axis, so phi is determined from
        # the angle in the yz plane.

        snapshot = hoomd.Snapshot(device.communicator)
        N = 4
        if snapshot.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            snapshot.configuration.box = box
            snapshot.particles.N = N
            snapshot.particles.types = particle_types
            # shift particle positions slightly in z so MPI tests pass
            snapshot.particles.position[:] = [
                [
                    0.0,
                    d * numpy.cos(phi_rad / 2),
                    d * numpy.sin(phi_rad / 2) + 0.1,
                ],
                [0.0, 0.0, 0.1],
                [d, 0.0, 0.1],
                [
                    d,
                    d * numpy.cos(phi_rad / 2),
                    -d * numpy.sin(phi_rad / 2) + 0.1,
                ],
            ]

            snapshot.dihedrals.N = 1
            snapshot.dihedrals.types = ['A-A-A-A']
            snapshot.dihedrals.typeid[0] = 0
            snapshot.dihedrals.group[0] = (0, 1, 2, 3)

        return snapshot

    return make_snapshot


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, force, energy',
                         dihedral_test_parameters)
def test_before_attaching(dihedral_cls, dihedral_args, params, force, energy):
    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params
    for key in params:
        potential.params['A-A-A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, force, energy',
                         dihedral_test_parameters)
def test_after_attaching(dihedral_snapshot_factory, simulation_factory,
                         dihedral_cls, dihedral_args, params, force, energy):
    snapshot = dihedral_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snapshot)

    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)
    for key in params:
        assert potential.params['A-A-A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, force, energy',
                         dihedral_test_parameters)
def test_forces_and_energies(dihedral_snapshot_factory, simulation_factory,
                             dihedral_cls, dihedral_args, params, force,
                             energy):
    phi_deg = 45
    phi_rad = phi_deg * (numpy.pi / 180)
    snapshot = dihedral_snapshot_factory(phi_deg=phi_deg)
    sim = simulation_factory(snapshot)

    # the dihedral angle is in yz plane, thus no force along x axis
    force_array = force * numpy.asarray(
        [0, numpy.sin(-phi_rad / 2),
         numpy.cos(-phi_rad / 2)])
    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    sim_energies = potential.energies
    sim_forces = potential.forces
    if sim.device.communicator.rank == 0:
        assert sum(sim_energies) == pytest.approx(energy, rel=1e-2, abs=1e-5)
        numpy.testing.assert_allclose(sim_forces[0],
                                      force_array,
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[1],
                                      -1 * force_array,
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[2],
                                      [0, -1 * force_array[1], force_array[2]],
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[3],
                                      [0, force_array[1], -1 * force_array[2]],
                                      rtol=1e-2,
                                      atol=1e-5)


@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, force, energy',
                         dihedral_test_parameters)
def test_kernel_parameters(dihedral_snapshot_factory, simulation_factory,
                           dihedral_cls, dihedral_args, params, force, energy):
    phi_deg = 45
    snapshot = dihedral_snapshot_factory(phi_deg=phi_deg)
    sim = simulation_factory(snapshot)

    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    autotuned_kernel_parameter_check(instance=potential,
                                     activate=lambda: sim.run(1))


# Test Logging
@pytest.mark.parametrize(
    'cls, expected_namespace, expected_loggables',
    zip((md.dihedral.Dihedral, md.dihedral.Periodic, md.dihedral.Table,
         md.dihedral.OPLS), itertools.repeat(('md', 'dihedral')),
        itertools.repeat(expected_loggable_params)))
def test_logging(cls, expected_namespace, expected_loggables):
    logging_check(cls, expected_namespace, expected_loggables)


# Test Pickling
@pytest.mark.parametrize('dihedral_cls, dihedral_args, params, force, energy',
                         dihedral_test_parameters)
def test_pickling(simulation_factory, dihedral_snapshot_factory, dihedral_cls,
                  dihedral_args, params, force, energy):
    phi_deg = 45
    snapshot = dihedral_snapshot_factory(phi_deg=phi_deg)
    sim = simulation_factory(snapshot)
    potential = dihedral_cls(**dihedral_args)
    potential.params['A-A-A-A'] = params

    pickling_check(potential)
    integrator = hoomd.md.Integrator(0.05, forces=[potential])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(potential)
