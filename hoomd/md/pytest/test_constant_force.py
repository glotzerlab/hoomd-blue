# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check, autotuned_kernel_parameter_check


def test_attributes():
    constant = hoomd.md.force.Constant(filter=hoomd.filter.All())

    assert constant.constant_force['A'] == (0.0, 0.0, 0.0)
    assert constant.constant_torque['A'] == (0.0, 0.0, 0.0)

    constant.constant_force['A'] = (0.5, 0.0, 0.0)
    assert constant.constant_force['A'] == (0.5, 0.0, 0.0)
    constant.constant_force['A'] = (0.0, 0.0, 1.0)
    assert constant.constant_force['A'] == (0.0, 0.0, 1.0)


def test_attach(simulation_factory, two_particle_snapshot_factory):
    constant = hoomd.md.force.Constant(filter=hoomd.filter.All())

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
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


def test_kernel_parameters(simulation_factory, two_particle_snapshot_factory):
    constant = hoomd.md.force.Constant(filter=hoomd.filter.All())

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(constant)
    sim.operations.integrator = integrator
    sim.run(0)

    autotuned_kernel_parameter_check(instance=constant,
                                     activate=lambda: sim.run(1))


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
