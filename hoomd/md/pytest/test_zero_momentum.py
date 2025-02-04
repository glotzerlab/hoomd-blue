# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import numpy as np


def test_before_attaching():
    trigger = hoomd.trigger.Periodic(100)
    zm = hoomd.md.update.ZeroMomentum(trigger)
    assert zm.trigger is trigger

    trigger = hoomd.trigger.Periodic(10, 30)
    zm.trigger = trigger
    assert zm.trigger is trigger


def test_after_attaching(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nve])

    trigger = hoomd.trigger.Periodic(100)
    zm = hoomd.md.update.ZeroMomentum(trigger)
    sim.operations.add(zm)
    sim.run(0)

    assert zm.trigger is trigger
    trigger = hoomd.trigger.Periodic(10, 30)
    zm.trigger = trigger
    assert zm.trigger is trigger

    sim.run(100)


def test_momentum_is_zero(simulation_factory, two_particle_snapshot_factory):
    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.velocity[0] = [0, 0, 0]
        snap.particles.velocity[1] = [2, 0, 0]
        snap.particles.mass[0] = 1
        snap.particles.mass[1] = 1
    sim = simulation_factory(snap)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nve])

    zm = hoomd.md.update.ZeroMomentum(hoomd.trigger.Periodic(1))
    sim.operations.add(zm)

    sim.run(1)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        masses = snap.particles.mass
        velocities = snap.particles.velocity
        for i in range(3):
            pi = sum([m * v[i] for m, v in zip(masses, velocities)])
            np.testing.assert_allclose(pi, 0, atol=1e-5)
