# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
from hoomd.conftest import pickling_check, autotuned_kernel_parameter_check


def test_attributes():
    active = hoomd.md.force.Active(filter=hoomd.filter.All())

    assert active.active_force['A'] == (1.0, 0.0, 0.0)
    assert active.active_torque['A'] == (0.0, 0.0, 0.0)

    active.active_force['A'] = (0.5, 0.0, 0.0)
    assert active.active_force['A'] == (0.5, 0.0, 0.0)
    active.active_force['A'] = (0.0, 0.0, 1.0)
    assert active.active_force['A'] == (0.0, 0.0, 1.0)


def test_attributes_constraints():
    plane = hoomd.md.manifold.Plane()
    active = hoomd.md.force.ActiveOnManifold(filter=hoomd.filter.All(),
                                             manifold_constraint=plane)

    assert active.active_force['A'] == (1.0, 0.0, 0.0)
    assert active.active_torque['A'] == (0.0, 0.0, 0.0)
    assert active.manifold_constraint == plane

    active.active_force['A'] = (0.5, 0.0, 0.0)
    assert active.active_force['A'] == (0.5, 0.0, 0.0)
    active.active_force['A'] = (0.0, 0.0, 1.0)
    assert active.active_force['A'] == (0.0, 0.0, 1.0)

    sphere = hoomd.md.manifold.Sphere(r=5)
    with pytest.raises(AttributeError):
        active.manifold_constraint = sphere
    assert active.manifold_constraint == plane


def test_attach(simulation_factory, two_particle_snapshot_factory):
    active = hoomd.md.force.Active(filter=hoomd.filter.All())

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    assert active.active_force['A'] == (1.0, 0.0, 0.0)
    assert active.active_torque['A'] == (0.0, 0.0, 0.0)

    active.active_force['A'] = (0.5, 0.0, 0.0)
    assert active.active_force['A'] == (0.5, 0.0, 0.0)
    active.active_force['A'] = (0.0, 0.0, 1.0)
    assert active.active_force['A'] == (0.0, 0.0, 1.0)


def test_kernel_parameters(simulation_factory, two_particle_snapshot_factory):
    active = hoomd.md.force.Active(filter=hoomd.filter.All())

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    autotuned_kernel_parameter_check(instance=active,
                                     activate=lambda: sim.run(1))


def test_attach_manifold(simulation_factory, two_particle_snapshot_factory):
    plane = hoomd.md.manifold.Plane()
    active = hoomd.md.force.ActiveOnManifold(filter=hoomd.filter.All(),
                                             manifold_constraint=plane)

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    assert active.active_force['A'] == (1.0, 0.0, 0.0)
    assert active.active_torque['A'] == (0.0, 0.0, 0.0)
    assert active.manifold_constraint == plane

    active.active_force['A'] = (0.5, 0.0, 0.0)
    assert active.active_force['A'] == (0.5, 0.0, 0.0)
    active.active_force['A'] = (0.0, 0.0, 1.0)
    assert active.active_force['A'] == (0.0, 0.0, 1.0)
    sphere = hoomd.md.manifold.Sphere(r=2)
    with pytest.raises(AttributeError):
        active.manifold_constraint = sphere
    assert active.manifold_constraint == plane


def test_kernel_parameters_manifold(simulation_factory,
                                    two_particle_snapshot_factory):
    plane = hoomd.md.manifold.Plane()
    active = hoomd.md.force.ActiveOnManifold(filter=hoomd.filter.All(),
                                             manifold_constraint=plane)

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(active)
    sim.operations.integrator = integrator
    sim.run(0)

    autotuned_kernel_parameter_check(instance=active,
                                     activate=lambda: sim.run(1))


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    active = hoomd.md.force.Active(filter=hoomd.filter.All())
    pickling_check(active)
    integrator = hoomd.md.Integrator(
        .05,
        methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0)],
        forces=[active])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(active)


def test_pickling_constraint(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    active = hoomd.md.force.ActiveOnManifold(
        filter=hoomd.filter.All(),
        manifold_constraint=hoomd.md.manifold.Plane())
    pickling_check(active)
    integrator = hoomd.md.Integrator(
        .05,
        methods=[hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0)],
        forces=[active])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(active)
