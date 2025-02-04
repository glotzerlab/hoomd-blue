# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
from hoomd.conftest import operation_pickling_check


def test_balance_properties():
    trigger = hoomd.trigger.Periodic(3)
    balance = hoomd.tune.LoadBalancer(trigger,
                                      x=True,
                                      y=True,
                                      z=True,
                                      tolerance=1.125,
                                      max_iterations=1)

    assert balance.trigger is trigger

    assert balance.x
    balance.x = False
    assert not balance.x

    assert balance.y
    balance.y = False
    assert not balance.y

    assert balance.z
    balance.z = False
    assert not balance.z

    assert balance.tolerance == 1.125
    balance.tolerance = 1.5
    assert balance.tolerance == 1.5

    assert balance.max_iterations == 1
    balance.max_iterations = 5
    assert balance.max_iterations == 5


def test_attach_detach(simulation_factory, lattice_snapshot_factory):
    snapshot = lattice_snapshot_factory()
    sim = simulation_factory(snapshot)
    trigger = hoomd.trigger.Periodic(3)

    balance = hoomd.tune.LoadBalancer(trigger,
                                      x=True,
                                      y=True,
                                      z=True,
                                      tolerance=1.125,
                                      max_iterations=1)

    sim.operations.tuners.append(balance)
    sim.run(0)

    assert balance.trigger is trigger

    assert balance.x
    balance.x = False
    assert not balance.x

    assert balance.y
    balance.y = False
    assert not balance.y

    assert balance.z
    balance.z = False
    assert not balance.z

    assert balance.tolerance == 1.125
    balance.tolerance = 1.5
    assert balance.tolerance == 1.5

    assert balance.max_iterations == 1
    balance.max_iterations = 5
    assert balance.max_iterations == 5

    sim.operations.tuners.remove(balance)


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    trigger = hoomd.trigger.Periodic(3)
    balance = hoomd.tune.LoadBalancer(trigger,
                                      x=True,
                                      y=True,
                                      z=True,
                                      tolerance=1.125,
                                      max_iterations=1)

    sim = simulation_factory(two_particle_snapshot_factory())
    operation_pickling_check(balance, sim)


def test_balance_action(device, simulation_factory, lattice_snapshot_factory):
    """Test that the load balancer does something."""
    if device.communicator.num_ranks != 2:
        pytest.skip("Test supports only 2 ranks")

    snapshot = lattice_snapshot_factory()

    # expand the box in one direction to make a non-uniform density distribution
    # this places all particles in the lower MPI domain
    box = list(snapshot.configuration.box)
    if snapshot.communicator.rank == 0:
        snapshot.particles.position[:, 2] -= box[2] / 2
    box[2] *= 2
    snapshot.configuration.box = box
    sim = simulation_factory(snapshot, domain_decomposition=(1, 1, 2))
    assert sim.state.domain_decomposition_split_fractions == ([], [], [0.5])

    balance = hoomd.tune.LoadBalancer(trigger=hoomd.trigger.Periodic(1))
    sim.operations.tuners.append(balance)
    sim.run(1)

    # the load balance should move the split place down toward the particles
    assert sim.state.domain_decomposition_split_fractions[2][0] < 0.5
