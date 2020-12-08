import hoomd
import pytest
import numpy
import itertools


def test_attach(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0, seed=1))
    integrator.forces.append(hoomd.md.force.Active(filter=hoomd.filter.All(), seed=2,rotation_diff=0.01))
    sim.operations.integrator = integrator
    sim.operations._schedule()
    sim.run(10)

