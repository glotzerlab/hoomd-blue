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

    # default value
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


@pytest.mark.serial
@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_after_attaching(attr, value, valid_args, simulation_factory,
                         lattice_snapshot_factory):
    integrator, args, n_dimensions = valid_args
    snap = lattice_snapshot_factory(particle_types=['A'],
                                    dimensions=n_dimensions)
    sim = simulation_factory(snap)

    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()
    mc = integrator()
    mc.shape["A"] = args
    sim.operations.add(mc)

    # default value
    assert getattr(mc, attr).value == 1.0

    sim.run(0)

    setattr(mc, attr, value)
    assert getattr(mc, attr) == value
