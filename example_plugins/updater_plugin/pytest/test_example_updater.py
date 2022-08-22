# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from typing import List, Optional
import itertools
import pytest

import hoomd
from hoomd import updater_plugin, operation
from hoomd.device import Device

import numpy as np


def build_system(
    vel: Optional[List[float]] = None,
    device: Optional[Device] = None
) -> hoomd.Simulation:
    """Build a system of one partice with velocity `vel`. `vel` should be a
    list of 3 floats, and is defaulted to [1, 1, 1]."""

    if vel is None:
        vel = [1, 1, 1]
    else:
        if not isinstance(vel, list):
            raise ValueError("`vel` must be a list")
        if len(vel) != 3:
            raise ValueError("`vel` must be a list of length 3")

    if device is None:
        device = hoomd.device.CPU()

    sim = hoomd.Simulation(device, 0)

    snap = hoomd.Snapshot()
    snap.particles.N = 1
    snap.particles.types = ["A"]
    snap.particles.position[:] = [[0, 0, 0]]
    snap.particles.velocity[:] = [vel]

    snap.configuration.box = [1, 1, 1, 0, 0, 0]

    sim.create_state_from_snapshot(snap)

    return sim


rng = np.random.default_rng(seed=0)
velocities = rng.random((5, 3))
devices = [hoomd.device.CPU()]
if hoomd.device.GPU.is_available():
    devices.append(hoomd.device.GPU())

testdata = list(itertools.product(velocities, devices))


@pytest.mark.parametrize("vel,device", testdata)
def test_updater(vel, device):

    sim = build_system(list(vel), device)

    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.NVE(hoomd.filter.All())
    integrator.methods = [nve]

    sim.operations.integrator = integrator

    updater: operation.Updater = updater_plugin.update.ExampleUpdater(
        hoomd.trigger.On(sim.timestep)
    )
    sim.operations.updaters.append(updater)

    sim.run(0)

    velocity = sim.state.get_snapshot().particles.velocity[0]
    np.testing.assert_array_almost_equal(
        velocity,
        vel,
        decimal=6
    )

    sim.run(1)

    velocity = sim.state.get_snapshot().particles.velocity[0]
    np.testing.assert_array_almost_equal(
        velocity,
        np.array([0.0, 0.0, 0.0]),
        decimal=6
    )
