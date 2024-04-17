# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

import hoomd
from hoomd.conftest import pickling_check


@pytest.fixture
def snap():
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 0
        snap.particles.types = ["A"]
        snap.mpcd.N = 1
        snap.mpcd.types = ["A"]
    return snap


class TestParticleSorter:

    def test_create(self, simulation_factory, snap):
        sim = simulation_factory(snap)

        sorter = hoomd.mpcd.tune.ParticleSorter(trigger=5)
        assert isinstance(sorter.trigger, hoomd.trigger.Trigger)

        trigger = hoomd.trigger.Periodic(50)
        sorter.trigger = trigger
        assert sorter.trigger is trigger

        ig = hoomd.mpcd.Integrator(dt=0.02, mpcd_particle_sorter=sorter)
        sim.operations.integrator = ig
        sim.run(0)
        assert sorter.trigger is trigger

    def test_pickling(self, simulation_factory, snap):
        sorter = hoomd.mpcd.tune.ParticleSorter(trigger=5)
        pickling_check(sorter)

        sorter.trigger = hoomd.trigger.Periodic(50)
        pickling_check(sorter)

        sim = simulation_factory(snap)
        sim.operations.integrator = hoomd.mpcd.Integrator(
            dt=0.02, mpcd_particle_sorter=sorter)
        sim.run(0)
        pickling_check(sorter)
