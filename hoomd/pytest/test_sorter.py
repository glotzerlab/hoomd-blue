# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test ParticleSorter."""

from hoomd.conftest import operation_pickling_check
import hoomd


def test_attributes():
    """Test ParticleSorter attributes before attaching."""
    trigger = hoomd.trigger.Periodic(500)
    sorter = hoomd.tune.ParticleSorter(trigger=trigger, grid=32)

    assert sorter.trigger is trigger
    assert sorter.grid == 32


def test_attributes_attached(simulation_factory, two_particle_snapshot_factory):
    """Test ParticleSorter attributes after attaching."""
    trigger = hoomd.trigger.Periodic(500)
    sorter = hoomd.tune.ParticleSorter(trigger=trigger, grid=32)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.tuners.append(sorter)
    sim.run(0)

    assert sorter.trigger is trigger
    assert sorter.grid == 32


def test_default_sorter(simulation_factory, two_particle_snapshot_factory):
    """Test that the default Simulation includes a ParticleSorter."""
    sim = simulation_factory(two_particle_snapshot_factory())

    assert len(sim.operations.tuners) == 1
    assert isinstance(sim.operations.tuners[0], hoomd.tune.ParticleSorter)


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    """Test pickling `hoomd.tune.ParticleSorter`."""
    sim = simulation_factory(two_particle_snapshot_factory())
    # need to remove tuner since operation_pickling_check adds operation to
    # simulation
    sorter = sim.operations.tuners.pop()
    operation_pickling_check(sorter, sim)
