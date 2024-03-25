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
angle_test_parameters = [
    (
        hoomd.md.angle.Harmonic,
        dict(),
        dict(k=3.0, t0=numpy.pi / 2),
        -1.5708,
        0.4112,
    ),
    (
        hoomd.md.angle.Harmonic,
        dict(),
        dict(k=10.0, t0=numpy.pi / 4),
        2.6180,
        0.3427,
    ),
    (
        hoomd.md.angle.Harmonic,
        dict(),
        dict(k=5.0, t0=numpy.pi / 6),
        2.6180,
        0.6854,
    ),
    (
        hoomd.md.angle.CosineSquared,
        dict(),
        dict(k=3.0, t0=numpy.pi / 2),
        -1.29904,
        0.375,
    ),
    (
        hoomd.md.angle.CosineSquared,
        dict(),
        dict(k=10.0, t0=numpy.pi / 4),
        1.7936,
        0.214466,
    ),
    (
        hoomd.md.angle.CosineSquared,
        dict(),
        dict(k=5.0, t0=numpy.pi / 6),
        1.58494,
        0.334936,
    ),
    (
        hoomd.md.angle.Table,
        dict(width=2),
        dict(U=[0, 10], tau=[0, 1]),
        -1 / 3,
        10 / 3,
    ),
]


@pytest.fixture(scope='session')
def triplet_snapshot_factory(device):

    def make_snapshot(d=1.0,
                      theta_deg=60,
                      particle_types=['A'],
                      dimensions=3,
                      L=20):
        theta_rad = theta_deg * (numpy.pi / 180)
        snapshot = hoomd.Snapshot(device.communicator)
        N = 3
        if snapshot.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            snapshot.configuration.box = box
            snapshot.particles.N = N

            base_positions = numpy.array([
                [
                    -d * numpy.sin(theta_rad / 2), d * numpy.cos(theta_rad / 2),
                    0.0
                ],
                [0.0, 0.0, 0.0],
                [
                    d * numpy.sin(theta_rad / 2),
                    d * numpy.cos(theta_rad / 2),
                    0.0,
                ],
            ])
            # move particles slightly in direction of MPI decomposition which
            # varies by simulation dimension
            nudge_dimension = 2 if dimensions == 3 else 1
            base_positions[:, nudge_dimension] += 0.1
            snapshot.particles.position[:] = base_positions
            snapshot.particles.types = particle_types
            snapshot.angles.N = 1
            snapshot.angles.types = ['A-A-A']
            snapshot.angles.typeid[0] = 0
            snapshot.angles.group[0] = (0, 1, 2)
        return snapshot

    return make_snapshot


@pytest.mark.parametrize('angle_cls, angle_args, params, force, energy',
                         angle_test_parameters)
def test_before_attaching(angle_cls, angle_args, params, force, energy):
    potential = angle_cls(**angle_args)
    potential.params['A-A-A'] = params
    for key in params:
        assert potential.params['A-A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize('angle_cls, angle_args, params, force, energy',
                         angle_test_parameters)
def test_after_attaching(triplet_snapshot_factory, simulation_factory,
                         angle_cls, angle_args, params, force, energy):
    sim = simulation_factory(triplet_snapshot_factory())

    potential = angle_cls(**angle_args)
    potential.params['A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)
    for key in params:
        assert potential.params['A-A-A'][key] == pytest.approx(params[key])


@pytest.mark.parametrize('angle_cls, angle_args, params, force, energy',
                         angle_test_parameters)
def test_forces_and_energies(triplet_snapshot_factory, simulation_factory,
                             angle_cls, angle_args, params, force, energy):
    theta_deg = 60
    theta_rad = theta_deg * (numpy.pi / 180)
    snapshot = triplet_snapshot_factory(theta_deg=theta_deg)
    sim = simulation_factory(snapshot)

    force_array = force * numpy.asarray(
        [numpy.cos(theta_rad / 2),
         numpy.sin(theta_rad / 2), 0])
    potential = angle_cls(**angle_args)
    potential.params['A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    sim_energy = potential.energy
    sim_forces = potential.forces
    if sim.device.communicator.rank == 0:
        assert sim_energy == pytest.approx(energy, rel=1e-2)
        numpy.testing.assert_allclose(sim_forces[0],
                                      force_array,
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(sim_forces[1], [0, -1 * force, 0],
                                      rtol=1e-2,
                                      atol=1e-5)
        numpy.testing.assert_allclose(
            sim_forces[2],
            [-1 * force_array[0], force_array[1], force_array[2]],
            rtol=1e-2,
            atol=1e-5)


@pytest.mark.parametrize('angle_cls, angle_args, params, force, energy',
                         angle_test_parameters)
def test_kernel_parameters(triplet_snapshot_factory, simulation_factory,
                           angle_cls, angle_args, params, force, energy):
    theta_deg = 60
    snapshot = triplet_snapshot_factory(theta_deg=theta_deg)
    sim = simulation_factory(snapshot)

    potential = angle_cls(**angle_args)
    potential.params['A-A-A'] = params

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    forces=[potential])

    sim.run(0)

    autotuned_kernel_parameter_check(instance=potential,
                                     activate=lambda: sim.run(1))


# Test Logging
@pytest.mark.parametrize(
    'cls, expected_namespace, expected_loggables',
    zip((md.angle.Angle, md.angle.Harmonic, md.angle.CosineSquared,
         md.angle.Table), itertools.repeat(('md', 'angle')),
        itertools.repeat(expected_loggable_params)))
def test_logging(cls, expected_namespace, expected_loggables):
    logging_check(cls, expected_namespace, expected_loggables)


# Test Pickling
@pytest.mark.parametrize('angle_cls, angle_args, params, force, energy',
                         angle_test_parameters)
def test_pickling(simulation_factory, triplet_snapshot_factory, angle_cls,
                  angle_args, params, force, energy):
    theta_deg = 60
    snapshot = triplet_snapshot_factory(theta_deg=theta_deg)
    sim = simulation_factory(snapshot)
    potential = angle_cls(**angle_args)
    potential.params['A-A-A'] = params

    pickling_check(potential)
    integrator = hoomd.md.Integrator(0.05, forces=[potential])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(potential)
