# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import expected_loggable_params
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)

import pytest
import numpy
import math

import itertools


class PeriodicImproperMath:
    # Class that contains the math for the periodic improper potential.

    @staticmethod
    def dchi_dr1(n1, n2, r1, r2, r3, r4):
        n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
        n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))
        numerator = numpy.dot(n1hat, n2hat) * \
            numpy.cross(numpy.dot(n1hat, n2hat) * n1hat - n2hat, r2 - r3) / \
            numpy.linalg.norm(n1)

        denominator = \
            numpy.sqrt(
                1 - numpy.dot(
                    numpy.cross(n1hat, n2hat),
                    numpy.cross(n1hat, n2hat)
                )
            ) \
            * numpy.linalg.norm(numpy.cross(n1hat, n2hat))

        return numerator / denominator

    @staticmethod
    def dchi_dr2(n1, n2, r1, r2, r3, r4):
        n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
        n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))

        numerator = numpy.dot(n1hat, n2hat) \
            * (
                numpy.cross(
                    numpy.dot(n1hat, n2hat) * n2hat - n1hat,
                    r3 - r4
                )
                / numpy.linalg.norm(n2) - numpy.cross(
                    numpy.dot(n1hat, n2hat) * n1hat - n2hat,
                    r1 - r3
                ) / numpy.linalg.norm(n1)
            )

        denominator = numpy.sqrt(
            1 - numpy.dot(
                numpy.cross(n1hat, n2hat),
                numpy.cross(n1hat, n2hat))
            ) \
            * numpy.linalg.norm(numpy.cross(n1hat, n2hat))

        return numerator / denominator

    @staticmethod
    def dchi_dr3(n1, n2, r1, r2, r3, r4):
        n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
        n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))

        numerator = numpy.dot(n1hat, n2hat) \
            * (
                numpy.cross(
                    numpy.dot(n1hat, n2hat) * n1hat - n2hat,
                    r1 - r2
                )
                / numpy.linalg.norm(n1) - numpy.cross(
                    numpy.dot(n1hat, n2hat) * n2hat - n1hat,
                    r2 - r4
                ) / numpy.linalg.norm(n2)
            )

        denominator = numpy.sqrt(
            1 - numpy.dot(
                numpy.cross(n1hat, n2hat),
                numpy.cross(n1hat, n2hat))
            ) \
            * numpy.linalg.norm(numpy.cross(n1hat, n2hat))

        return numerator / denominator

    @staticmethod
    def dchi_dr4(n1, n2, r1, r2, r3, r4):
        n1hat = n1 / numpy.sqrt(numpy.dot(n1, n1))
        n2hat = n2 / numpy.sqrt(numpy.dot(n2, n2))

        numerator = numpy.dot(n1hat, n2hat) * \
            numpy.cross(
                numpy.dot(n1hat, n2hat) * n2hat - n1hat,
                r2 - r3
            ) / numpy.linalg.norm(n2)

        denominator = \
            numpy.sqrt(
                1 - numpy.dot(
                    numpy.cross(n1hat, n2hat),
                    numpy.cross(n1hat, n2hat)
                )
            ) \
            * numpy.linalg.norm(
                numpy.cross(n1hat, n2hat)
            )

        return numerator / denominator

    @staticmethod
    def chi_from_pos(posa, posb, posc, posd):
        n1 = numpy.cross(posa - posb, posb - posc)
        n2 = numpy.cross(posb - posc, posc - posd)
        mag = numpy.dot(n1, n2) / numpy.linalg.norm(n1) / numpy.linalg.norm(n2)
        return math.acos(numpy.linalg.norm(mag))

    @staticmethod
    def du_dchi_periodic(chi, chi0, k, n, d):
        return -k * n * d * numpy.sin(n * chi - chi0)

    @staticmethod
    def du_dchi_harmonic(chi, k, chi0):
        return k * (chi - chi0)

    @staticmethod
    def periodic_improper_energy(chi, k, n, d, chi0):
        return (k * (1 + d * numpy.cos(n * chi - chi0)))

    @staticmethod
    def get_force_vectors(chi, n1, n2, r1, r2, r3, r4, chi0, k, d, n):
        f_matrix = numpy.zeros((4, 3))
        f_matrix[0, :] = PeriodicImproperMath.dchi_dr1(
            n1, n2, r1, r2, r3, r4) * PeriodicImproperMath.du_dchi_periodic(
                chi, chi0=chi0, k=k, d=d, n=n)
        f_matrix[1, :] = PeriodicImproperMath.dchi_dr2(
            n1, n2, r1, r2, r3, r4) * PeriodicImproperMath.du_dchi_periodic(
                chi, chi0=chi0, k=k, d=d, n=n)
        f_matrix[2, :] = PeriodicImproperMath.dchi_dr3(
            n1, n2, r1, r2, r3, r4) * PeriodicImproperMath.du_dchi_periodic(
                chi, chi0=chi0, k=k, d=d, n=n)
        f_matrix[3, :] = PeriodicImproperMath.dchi_dr4(
            n1, n2, r1, r2, r3, r4) * PeriodicImproperMath.du_dchi_periodic(
                chi, chi0=chi0, k=k, d=d, n=n)
        return f_matrix


# Test parameters include the class, improper params, force, and energy.
# This is parameterized to plan for any future expansion with additional
# improper potentials.

pos = numpy.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0.1],
])

# plane 1 normal vector
n1 = numpy.cross(pos[0, :] - pos[1, :], pos[1, :] - pos[2, :])
# plane 2 normal vector
n2 = numpy.cross(pos[1, :] - pos[2, :], pos[2, :] - pos[3, :])
# improper angle
chi = PeriodicImproperMath.chi_from_pos(posa=pos[0, :],
                                        posb=pos[1, :],
                                        posc=pos[2, :],
                                        posd=pos[3, :])

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
    (hoomd.md.improper.Periodic, dict(k=3.0, d=-1, n=2, chi0=numpy.pi / 2),
     PeriodicImproperMath.get_force_vectors(chi,
                                            n1=n1,
                                            n2=n2,
                                            r1=pos[0, :],
                                            r2=pos[1, :],
                                            r3=pos[2, :],
                                            r4=pos[3, :],
                                            chi0=numpy.pi / 2,
                                            k=3.0,
                                            d=-1,
                                            n=2) / 2,
     PeriodicImproperMath.periodic_improper_energy(
         chi, k=3.0, d=-1, n=2, chi0=numpy.pi / 2) / 2),
    (hoomd.md.improper.Periodic, dict(k=10.0, d=1, n=1, chi0=numpy.pi / 4),
     PeriodicImproperMath.get_force_vectors(chi,
                                            n1=n1,
                                            n2=n2,
                                            r1=pos[0, :],
                                            r2=pos[1, :],
                                            r3=pos[2, :],
                                            r4=pos[3, :],
                                            chi0=numpy.pi / 4,
                                            k=10.0,
                                            d=1,
                                            n=1) / 2,
     PeriodicImproperMath.periodic_improper_energy(
         chi, k=10.0, d=1, n=1, chi0=numpy.pi / 4) / 2),
    (hoomd.md.improper.Periodic, dict(k=5.0, d=1, n=3, chi0=numpy.pi / 6),
     PeriodicImproperMath.get_force_vectors(chi,
                                            n1=n1,
                                            n2=n2,
                                            r1=pos[0, :],
                                            r2=pos[1, :],
                                            r3=pos[2, :],
                                            r4=pos[3, :],
                                            chi0=numpy.pi / 6,
                                            k=5.0,
                                            d=1,
                                            n=3) / 2,
     PeriodicImproperMath.periodic_improper_energy(
         chi, k=5.0, d=1, n=3, chi0=numpy.pi / 6) / 2)
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
