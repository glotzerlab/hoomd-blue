# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test that `HalfStepHook` works."""

import numpy as np
import pytest
import hoomd

from hoomd import md


class DistanceCV(md.HalfStepHook):

    def __init__(self, sim):
        md.HalfStepHook.__init__(self)
        self.state = sim.state
        self.time_series = []

    def update(self, _):
        snapshot = self.state.get_snapshot()
        if snapshot.communicator.rank == 0:
            r1, r2 = snapshot.particles.position
            self.time_series.append(np.linalg.norm(r1 - r2))


@pytest.fixture
def make_simulation(simulation_factory, two_particle_snapshot_factory):

    def sim_factory(particle_types=['A'], dimensions=3, d=1, L=20):
        snap = two_particle_snapshot_factory()
        if snap.communicator.rank == 0:
            snap.constraints.N = 1
            snap.constraints.value[0] = 1.0
            snap.constraints.group[0] = [0, 1]
        return simulation_factory(snap)

    return sim_factory


@pytest.fixture
def integrator_elements():
    nlist = md.nlist.Cell(buffer=0.4)
    lj = md.pair.LJ(nlist=nlist, default_r_cut=2.5)
    gauss = md.pair.Gaussian(nlist, default_r_cut=3.0)
    lj.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
    gauss.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
    return {
        "methods": [md.methods.ConstantVolume(hoomd.filter.All())],
        "forces": [lj, gauss],
        "constraints": [md.constrain.Distance()]
    }


def test_half_step_hook(make_simulation, integrator_elements):
    sim = make_simulation()
    half_step_hook = DistanceCV(sim)
    integrator = md.Integrator(0.005, **integrator_elements)
    integrator.half_step_hook = half_step_hook
    sim.operations.integrator = integrator
    sim.run(1)
    if sim.device.communicator.rank == 0:
        assert len(half_step_hook.time_series) == 1
    sim.run(9)
    if sim.device.communicator.rank == 0:
        assert len(half_step_hook.time_series) == 10
