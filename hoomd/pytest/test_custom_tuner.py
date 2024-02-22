# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd


class ModifyTrigger(hoomd.custom.Action):
    test_attr = True
    trigger = None  # should not be passed through

    def attach(self, simulation):
        self._simulation = simulation

    def act(self, timestep):
        # The particle sorter
        self._simulation.operations.tuners[0].trigger = timestep


def test_custom_tuner_passthrough():
    tuner = hoomd.tune.CustomTuner(1, ModifyTrigger())
    assert tuner.test_attr
    assert tuner.trigger is not None
    assert tuner.trigger == hoomd.trigger.Periodic(1)


def test_custom_tuner_act(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.run(5)
    sim.operations += hoomd.tune.CustomTuner(1, ModifyTrigger())
    sim.run(1)
    assert sim.operations.tuners[0].trigger.period == 5
