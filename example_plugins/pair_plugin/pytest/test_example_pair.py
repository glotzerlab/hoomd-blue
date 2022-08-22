# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from typing import List, Optional, Tuple
import itertools
import pytest

import hoomd
from hoomd.device import Device

from hoomd import pair_plugin

import numpy as np
from numpy.typing import NDArray


def build_binary_system(pos: Optional[List[float]] = None,
                        device: Optional[Device] = None) -> hoomd.Simulation:
    """Builds a two-particle system.

    Builds a system of two particles, with one particle located at the origin
    and the other located at `pos`. `pos` should be a list of 3 floats, and is
    defaulted to [0.5, 0, 0]. The box is 4x4x4.
    """
    if pos is None:
        pos = [0.5, 0, 0]
    else:
        if not isinstance(pos, list):
            raise ValueError("`pos` must be a list")
        if len(pos) != 3:
            raise ValueError("`pos` must be a list of length 3")

    if device is None:
        device = hoomd.device.CPU()

    sim = hoomd.Simulation(device, 0)

    snap = hoomd.Snapshot()
    snap.particles.N = 2
    snap.particles.types = ["A"]
    snap.particles.position[:] = [[0, 0, 0], pos]
    snap.configuration.box = [4, 4, 4, 0, 0, 0]

    sim.create_state_from_snapshot(snap)

    return sim


def harm_force_and_energy(
        dx: List[float],
        k: float,
        sigma: float,
        r_cut: float,
        shift: Optional[bool] = False) -> Tuple[NDArray[np.float64], float]:

    dr = np.linalg.norm(dx)

    if dr >= r_cut:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0

    f = -k * (dr - sigma) * np.array(dx, dtype=np.float64) / dr
    e = 0.5 * k * (dr - sigma)**2
    if shift:
        e -= 0.5 * k * (r_cut - sigma)**2

    return f, e


rng = np.random.default_rng(seed=0)
positions = 0.5 * rng.random((10, 3)) - 0.25
positions = list(positions) + list(2.0 * rng.random((10, 3)) - 1.0)
devices = [hoomd.device.CPU()]
if hoomd.device.GPU.is_available():
    devices.append(hoomd.device.GPU())
ks = [0.5, 1.0, 2.0, 5.0]
sigmas = [0.5, 1.0, 1.5]

testdata = list(itertools.product(positions, devices, ks, sigmas))


@pytest.mark.parametrize("pos,device,k,sigma", testdata)
def test_force_and_energy_eval(pos, device, k, sigma):

    sim = build_binary_system(list(pos), device)

    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.NVE(hoomd.filter.All())

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    example_pair: hoomd.md.pair.Pair = pair_plugin.pair.ExamplePair(
        cell, default_r_cut=sigma)
    example_pair.params[("A", "A")] = dict(k=k, sigma=sigma)
    integrator.forces = [example_pair]
    integrator.methods = [nve]

    sim.operations.integrator = integrator

    sim.run(0)

    f, e = harm_force_and_energy(pos, k, sigma, sigma)
    e /= 2.0

    forces = example_pair.forces
    print(forces, f)
    np.testing.assert_array_almost_equal(forces, [-f, f])

    energies = example_pair.energies
    print(energies, e)
    np.testing.assert_array_almost_equal(energies, [e, e])
