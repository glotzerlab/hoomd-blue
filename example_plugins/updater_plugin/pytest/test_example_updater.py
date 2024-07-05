# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

# Import the plugin module.
from hoomd import updater_plugin

# Import the hoomd Python package.
import hoomd
from hoomd import operation

import itertools
import pytest
import numpy as np

# Generate a list of velocities to test against. Hard-coded values are also
# appropriate here.
v_comp = np.linspace(-5.0, 5.0, 3)

# An array of 3-tuples that will be our testing velocities.
velocities = list(itertools.product(v_comp, v_comp, v_comp))


# Use pytest decorator to automate testing over the sequence of parameters.
@pytest.mark.parametrize("vel", velocities)
def test_updater(simulation_factory, one_particle_snapshot_factory, vel):

    # `one_particle_snapshot_factory` and `simulation_factory` are pytest
    # fixtures defined in hoomd/conftest.py. These factories automatically
    # handle iterating tests over different CPU and GPU devices.
    snap = one_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.velocity[0] = vel
    sim = simulation_factory(snap)

    # Add our plugin to the simulation.
    updater: operation.Updater = updater_plugin.update.ExampleUpdater(
        hoomd.trigger.On(sim.timestep))
    sim.operations.updaters.append(updater)

    # Test that the initial velocity matches our input.
    sim.run(0)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        velocity = snap.particles.velocity[0]
        np.testing.assert_array_almost_equal(velocity, vel, decimal=6)

    # Test that the velocity is properly zeroed after the update.
    sim.run(1)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        velocity = snap.particles.velocity[0]
        np.testing.assert_array_almost_equal(velocity,
                                             np.array([0.0, 0.0, 0.0]),
                                             decimal=6)
