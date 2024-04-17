# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np

import hoomd
from hoomd.conftest import pickling_check


def test_block_force(simulation_factory):
    # make the force
    force = hoomd.mpcd.force.BlockForce(force=2.0, separation=6.0, width=1.0)
    assert force.force == 2.0
    assert force.separation == 6.0
    assert force.width == 1.0
    pickling_check(force)

    # try changing the values
    force.force = 1.0
    force.separation = 4.0
    force.width = 0.5
    assert force.force == 1.0
    assert force.separation == 4.0
    assert force.width == 0.5

    # now attach and check again
    sim = simulation_factory()
    force._attach(sim)
    assert force.force == 1.0
    assert force.separation == 4.0
    assert force.width == 0.5

    # change values while attached
    force.force = 2.0
    force.separation = 6.0
    force.width = 1.0
    assert force.force == 2.0
    assert force.separation == 6.0
    assert force.width == 1.0
    pickling_check(force)


def test_constant_force(simulation_factory):
    # make the force
    force = hoomd.mpcd.force.ConstantForce(force=(1, -2, 3))
    np.testing.assert_array_almost_equal(force.force, (1, -2, 3))
    pickling_check(force)

    # try changing the force vector
    force.force = (-1, 2, -3)
    np.testing.assert_array_almost_equal(force.force, (-1, 2, -3))

    # now attach and check again
    sim = simulation_factory()
    force._attach(sim)
    np.testing.assert_array_almost_equal(force.force, (-1, 2, -3))

    # change values while attached
    force.force = (1, -2, 3)
    np.testing.assert_array_almost_equal(force.force, (1, -2, 3))
    pickling_check(force)


def test_sine_force(simulation_factory):
    # make the force
    force = hoomd.mpcd.force.SineForce(amplitude=2.0, wavenumber=1)
    assert force.amplitude == 2.0
    assert force.wavenumber == 1.0
    pickling_check(force)

    # try changing the values
    force.amplitude = 1.0
    force.wavenumber = 2.0
    assert force.amplitude == 1.0
    assert force.wavenumber == 2.0

    # now attach and check again
    sim = simulation_factory()
    force._attach(sim)
    assert force.amplitude == 1.0
    assert force.wavenumber == 2.0

    # change values while attached
    force.amplitude = 2.0
    force.wavenumber = 1.0
    assert force.amplitude == 2.0
    assert force.wavenumber == 1.0
    pickling_check(force)
