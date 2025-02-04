# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

# Import the plugin module.
from hoomd import pair_plugin

# Import the hoomd Python package.
import hoomd

import itertools
import pytest
import numpy as np


# Python implementation of the pair force and energy.
def harm_force_and_energy(dx, k, sigma, r_cut, shift=False):

    dr = np.linalg.norm(dx)

    if dr >= r_cut:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0

    f = k * (sigma - dr) * np.array(dx, dtype=np.float64) / dr
    e = 0.5 * k * (sigma - dr)**2
    if shift:
        e -= 0.5 * k * (r_cut - sigma)**2

    return f, e


# Build up list of parameters.
distances = np.linspace(0.1, 2.0, 3)
ks = [0.5, 2.0, 5.0]
sigmas = [0.5, 1.0, 1.5]
# No need to test "xplor", as that is handled outside of the plugin impl.
modes = ["none", "shift"]

testdata = list(itertools.product(distances, ks, sigmas, modes))


@pytest.mark.parametrize("distance, k, sigma, mode", testdata)
def test_force_and_energy_eval(simulation_factory,
                               two_particle_snapshot_factory, distance, k,
                               sigma, mode):

    # Build the simulation from the factory fixtures defined in
    # hoomd/conftest.py.
    sim = simulation_factory(two_particle_snapshot_factory(d=distance))

    # Setup integrator and force.
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.ConstantVolume(hoomd.filter.All())

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    example_pair: hoomd.md.pair.Pair = pair_plugin.pair.ExamplePair(
        cell, default_r_cut=sigma, mode=mode)
    example_pair.params[("A", "A")] = dict(k=k, sigma=sigma)
    integrator.forces = [example_pair]
    integrator.methods = [nve]

    sim.operations.integrator = integrator

    sim.run(0)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        vec_dist = snap.particles.position[1] - snap.particles.position[0]

        # Compute force and energy from Python
        shift = mode == "shift"
        f, e = harm_force_and_energy(vec_dist, k, sigma, sigma, shift)
        e /= 2.0

    # Test that the forces and energies match that predicted by the Python
    # implementation.
    forces = example_pair.forces
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(forces, [-f, f], decimal=6)

    energies = example_pair.energies
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(energies, [e, e], decimal=6)
