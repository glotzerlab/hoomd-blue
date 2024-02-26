# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.update.RemoveDrift."""

import hoomd
from hoomd.conftest import operation_pickling_check
import pytest
import numpy as np

# note: The parameterized tests validate parameters so we can't pass in values
# here that require preprocessing
valid_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10),
         reference_positions=[(0, 0, 0), (1, 0, 1)]),
    dict(trigger=hoomd.trigger.After(10),
         reference_positions=[(0, 0, 0), (1, 0, 1)]),
    dict(trigger=hoomd.trigger.Before(10),
         reference_positions=[(0, 0, 0), (1, 0, 1)])
]

valid_attrs = [('trigger', hoomd.trigger.Periodic(10000)),
               ('trigger', hoomd.trigger.After(100)),
               ('trigger', hoomd.trigger.Before(12345)),
               ('reference_positions', [(0, 0, 0), (1, 0, 1)])]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(constructor_args):
    """Test that RemoveDrift can be constructed with valid arguments."""
    remove_drift = hoomd.update.RemoveDrift(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(remove_drift, attr) == value)


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args):
    """Test that RemoveDrift can be attached with valid arguments."""
    remove_drift = hoomd.update.RemoveDrift(**constructor_args)
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'], d=2, L=50))
    sim.operations.updaters.append(remove_drift)

    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(remove_drift, attr) == value)


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(attr, value):
    """Test that RemoveDrift can get and set attributes."""
    remove_drift = hoomd.update.RemoveDrift(trigger=hoomd.trigger.Periodic(10),
                                            reference_positions=[(0, 0, 1),
                                                                 (-1, 0, 1)])

    setattr(remove_drift, attr, value)
    assert np.all(getattr(remove_drift, attr) == value)


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(attr, value, simulation_factory,
                                two_particle_snapshot_factory):
    """Test that RemoveDrift can get and set attributes while attached."""
    remove_drift = hoomd.update.RemoveDrift(trigger=hoomd.trigger.Periodic(10),
                                            reference_positions=[(0, 0, 1),
                                                                 (-1, 0, 1)])
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'], L=50))
    sim.operations.updaters.append(remove_drift)

    sim.run(0)

    setattr(remove_drift, attr, value)
    assert np.all(getattr(remove_drift, attr) == value)


def test_remove_drift(simulation_factory, lattice_snapshot_factory):
    """Test that RemoveDrift modifies positions correctly."""
    # reference positions in a simple cubic lattice with a=1
    reference_positions = [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5],
                           [0.5, -0.5, -0.5], [0.5, -0.5,
                                               0.5], [-0.5, 0.5, -0.5],
                           [-0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]]

    # initialize simulation with randomized positions (off lattice)
    snap = lattice_snapshot_factory(dimensions=3, n=2, a=1, r=0.1)
    sim = simulation_factory(snap)

    # add remove drift updater and run
    remove_drift = hoomd.update.RemoveDrift(
        trigger=hoomd.trigger.Periodic(1),
        reference_positions=reference_positions)
    sim.operations.updaters.append(remove_drift)
    sim.run(1)

    # ensure the drift is close to zero after the updater has been executed
    s = sim.state.get_snapshot()
    if s.communicator.rank == 0:
        new_positions = s.particles.position
        drift = np.mean(new_positions - reference_positions, axis=0)
        assert np.allclose(drift, [0, 0, 0])


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    """Test that RemoveDrift objects are picklable."""
    sim = simulation_factory(two_particle_snapshot_factory())
    remove_drift = hoomd.update.RemoveDrift(trigger=hoomd.trigger.Periodic(5),
                                            reference_positions=[(0, 0, 1),
                                                                 (-1, 0, 1)])
    operation_pickling_check(remove_drift, sim)
