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

