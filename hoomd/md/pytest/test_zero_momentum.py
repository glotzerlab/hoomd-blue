import hoomd
import numpy as np
import pytest

def test_before_attaching():
    trigger = hoomd.trigger.Periodic(100)
    zm = hoomd.md.update.ZeroMomentum(trigger)
    assert zm.trigger is trigger

    trigger = hoomd.trigger.Periodic(10, 30)
    zm.trigger = trigger
    assert zm.trigger is trigger


def test_after_attaching(simulation_factory,
                         two_particle_snapshot_factory):
    brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(),
                                         kT=hoomd.variant.Constant(2.0),
                                         seed=2)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[brownian])

    trigger = hoomd.trigger.Periodic(100)
    zm = hoomd.md.update.ZeroMomentum(trigger)
    sim.operations.add(zm)
    sim.run(0)

    assert zm.trigger is trigger
    trigger = hoomd.trigger.Periodic(10, 30)
    zm.trigger = trigger
    assert zm.trigger is trigger

    sim.run(100)


def test_momentum_is_zero(simulation_factory,
                          two_particle_snapshot_factory):
    snap = two_particle_snapshot_factory()
    if snap.exists:
        snap.particles.velocity[0] = [0, 0, 0]
        snap.particles.velocity[1] = [2, 0, 0]
        snap.particles.mass[0] = 1
        snap.particles.mass[1] = 1
    sim = simulation_factory(snap)
    brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(),
                                         kT=hoomd.variant.Constant(0.2),
                                         seed=2)
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[brownian])

    zm = hoomd.md.update.ZeroMomentum(hoomd.trigger.Periodic(1, 1))
    sim.operations.add(zm)
    sim.run(0)
    masses = sim.state.snapshot.particles.mass
    velocities = sim.state.snapshot.particles.velocity
    system_momentum = sum([np.linalg.norm(masses[i] * velocities[i]) for i in [0, 1]])
    assert system_momentum == 2

    sim.run(10)
    masses = sim.state.snapshot.particles.mass
    velocities = sim.state.snapshot.particles.velocity
    print(velocities)
    for j in [0, 1, 2]:
        j_momentum = sum([masses[i] * velocities[i][j] for i in [0, 1]])
        assert j_momentum == 0
