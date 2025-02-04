# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check, autotuned_kernel_parameter_check
import numpy


def test_attributes():
    constant = hoomd.md.force.Constant(filter=hoomd.filter.All())

    assert constant.constant_force['A'] == (0.0, 0.0, 0.0)
    assert constant.constant_torque['A'] == (0.0, 0.0, 0.0)

    constant.constant_force['A'] = (0.5, 0.0, 0.0)
    assert constant.constant_force['A'] == (0.5, 0.0, 0.0)
    constant.constant_force['A'] = (0.0, 0.0, 1.0)
    assert constant.constant_force['A'] == (0.0, 0.0, 1.0)


def test_attach_and_filter(simulation_factory, two_particle_snapshot_factory):
    constant = hoomd.md.force.Constant(filter=hoomd.filter.Type(['A']))

    snapshot = two_particle_snapshot_factory(particle_types=['A', 'B'],
                                             dimensions=3,
                                             d=8)
    if snapshot.communicator.rank == 0:
        snapshot.particles.typeid[:] = [1, 0]

    sim = simulation_factory(snapshot)

    integrator = hoomd.md.Integrator(0.0)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(constant)
    sim.operations.integrator = integrator
    sim.run(0)

    assert constant.constant_force['A'] == (0.0, 0.0, 0.0)
    assert constant.constant_torque['A'] == (0.0, 0.0, 0.0)

    constant.constant_force['A'] = (0.5, 0.0, 0.0)
    assert constant.constant_force['A'] == (0.5, 0.0, 0.0)
    constant.constant_torque['A'] = (0.0, 0.0, 1.0)
    assert constant.constant_torque['A'] == (0.0, 0.0, 1.0)

    constant.constant_force['B'] = (0.0, 0.125, 5.0)
    assert constant.constant_force['B'] == (0.0, 0.125, 5.0)
    constant.constant_torque['B'] = (4.0, -6.0, 0.5)
    assert constant.constant_torque['B'] == (4.0, -6.0, 0.5)

    sim.run(1)

    forces = constant.forces
    torques = constant.torques

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_equal(forces,
                                         [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        numpy.testing.assert_array_equal(torques,
                                         [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_types(simulation_factory, two_particle_snapshot_factory):
    constant = hoomd.md.force.Constant(filter=hoomd.filter.All())

    snapshot = two_particle_snapshot_factory(particle_types=['A', 'B'],
                                             dimensions=3,
                                             d=8)
    if snapshot.communicator.rank == 0:
        snapshot.particles.typeid[:] = [1, 0]

    sim = simulation_factory(snapshot)

    integrator = hoomd.md.Integrator(0.0)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(constant)
    sim.operations.integrator = integrator
    sim.run(0)

    constant.constant_force['A'] = (0.5, 0.0, 0.0)
    constant.constant_torque['A'] = (0.0, 0.0, 1.0)

    constant.constant_force['B'] = (0.0, 0.125, 5.0)
    constant.constant_torque['B'] = (4.0, -6.0, 0.5)

    sim.run(1)

    forces = constant.forces
    torques = constant.torques

    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_equal(forces,
                                         [[0.0, 0.125, 5.0], [0.5, 0.0, 0.0]])
        numpy.testing.assert_array_equal(torques,
                                         [[4.0, -6.0, 0.5], [0.0, 0.0, 1.0]])


def test_kernel_parameters(simulation_factory, two_particle_snapshot_factory):
    constant = hoomd.md.force.Constant(filter=hoomd.filter.All())

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(constant)
    sim.operations.integrator = integrator
    sim.run(0)

    def activate_kernel():
        constant.constant_force['A'] = (1.0, 2.0, 3.0)
        sim.run(1)

    autotuned_kernel_parameter_check(instance=constant,
                                     activate=activate_kernel)


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    constant = hoomd.md.force.Constant(filter=hoomd.filter.All())
    pickling_check(constant)
    integrator = hoomd.md.Integrator(
        .05,
        methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0)],
        forces=[constant])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(constant)
