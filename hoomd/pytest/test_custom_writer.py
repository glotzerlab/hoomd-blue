# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd


class WriteTimestep(hoomd.custom.Action):
    test_attr = True
    trigger = None  # should not be passed through

    def __init__(self):
        self.timesteps_run = []

    def act(self, timestep):
        self.timesteps_run.append(timestep)


def test_custom_writer_passthrough():
    writer = hoomd.update.CustomUpdater(1, WriteTimestep())
    assert writer.test_attr
    assert writer.trigger is not None


def test_custom_writer_act(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    writer = hoomd.write.CustomWriter(2, WriteTimestep())
    sim.operations += writer
    sim.run(10)
    assert writer.timesteps_run == [2, 4, 6, 8, 10]
