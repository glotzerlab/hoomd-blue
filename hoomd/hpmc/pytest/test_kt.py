# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test kT."""

import hoomd
import pytest
import hoomd.hpmc.pytest.conftest

# note: The parameterized tests validate parameters so we can't pass in values
# here that require preprocessing

valid_attrs = [
    ('kT', hoomd.variant.Constant(10)),
    ('kT', hoomd.variant.Ramp(1, 5, 0, 100)),
    ('kT', hoomd.variant.Cycle(1, 5, 0, 10, 20, 10, 15)),
    ('kT', hoomd.variant.Power(1, 5, 3, 0, 100)),
]


@pytest.mark.serial
@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(device, attr, value):
    """Test that kT can get and set attributes."""
    mc = hoomd.hpmc.integrate.Sphere()

    setattr(mc, attr, 1.0)
    assert getattr(mc, attr).value == 1.0

    setattr(mc, attr, value)
    assert getattr(mc, attr) == value


@pytest.mark.serial
@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(attr, value, simulation_factory,
                                two_particle_snapshot_factory):
    """Test that integrator can get and set attributes while attached."""
    sim = simulation_factory(two_particle_snapshot_factory())

    # BoxMC requires an HPMC integrator
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    setattr(mc, attr, value)
    assert getattr(mc, attr) == value
