# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd


class ChargeModifier(hoomd.custom.Action):
    test_attr = True
    trigger = None  # should not be passed through

    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            index = snap.particles.rtag[timestep % self._state.N_particles]
            if index < len(snap.particles.position):
                snap.particles.charge[index] = -5


def test_custom_updater_passthrough():
    updater = hoomd.update.CustomUpdater(1, ChargeModifier())
    assert updater.test_attr
    assert updater.trigger is not None
    assert updater.trigger == hoomd.trigger.Periodic(1)


def test_custom_updater_act(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations += hoomd.update.CustomUpdater(1, ChargeModifier())
    sim.run(1)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        assert snap.particles.charge[0] == -5
        assert snap.particles.charge[1] == 0
    sim.run(1)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        assert snap.particles.charge[0] == -5
        assert snap.particles.charge[1] == -5
