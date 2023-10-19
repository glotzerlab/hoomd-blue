# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import numpy as np
import pytest


@pytest.fixture
def small_snap():
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [20, 20, 20, 0, 0, 0]
        snap.particles.N = 1
        snap.particles.types = ["A"]
        snap.mpcd.N = 1
        snap.mpcd.types = ["A"]
    return snap


def test_create(small_snap, simulation_factory):
    sim = simulation_factory(small_snap)
    ig = hoomd.mpcd.Integrator(dt=0.02)
    sim.operations.integrator = ig
    sim.run(0)

    assert ig.cell_list is not None
    assert ig.streaming_method is None
    assert ig.collision_method is None
