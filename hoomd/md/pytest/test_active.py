import hoomd
import pytest
import numpy
import itertools


def test_attach(simulation_factory, two_particle_snapshot_factory, capsys):
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=8))
    integrator = hoomd.md.Integrator(.05)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=0))
    integrator.forces.append(hoomd.md.force.Active(filter=hoomd.filter.All(), rotation_diff=0.01))
    sim.operations.integrator = integrator
    sim.operations._schedule()
    sim.run(10)

    # test the seed warning is issued
    captured = capsys.readouterr()
    assert captured.err == "*Warning*: Simulation.seed is not set, using " \
                           "default seed=0\n" \
                           "*Warning*: Simulation.seed is not set, using " \
                           "default seed=0\n"
