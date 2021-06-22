# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.update.RemoveDrift."""

import hoomd
from hoomd.conftest import operation_pickling_check
import pytest
import hoomd.hpmc.pytest.conftest
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


@pytest.mark.cpu
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that RemoveDrift can be constructed with valid arguments."""
    cl = hoomd.hpmc.update.RemoveDrift(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(cl, attr) == value)


@pytest.mark.cpu
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(device, simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args, valid_args):
    """Test that RemoveDrift can be attached with valid arguments."""
    integrator = valid_args[0]
    args = valid_args[1]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"]
    mc = integrator()
    mc.shape["A"] = args
    mc.shape["B"] = args

    cl = hoomd.hpmc.update.RemoveDrift(**constructor_args)
    dim = 2 if 'polygon' in integrator.__name__.lower() else 3
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=dim,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(cl)
    sim.operations.integrator = mc

    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(cl, attr) == value)


@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(device, attr, value):
    """Test that RemoveDrift can get and set attributes."""
    cl = hoomd.hpmc.update.RemoveDrift(trigger=hoomd.trigger.Periodic(10),
                                       reference_positions=[(0, 0, 1),
                                                            (-1, 0, 1)])

    setattr(cl, attr, value)
    assert np.all(getattr(cl, attr) == value)


@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(device, attr, value, simulation_factory,
                                two_particle_snapshot_factory, valid_args):
    """Test that RemoveDrift can get and set attributes while attached."""
    integrator = valid_args[0]
    args = valid_args[1]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"]
    mc = integrator()
    mc.shape["A"] = args
    mc.shape["B"] = args

    cl = hoomd.hpmc.update.RemoveDrift(trigger=hoomd.trigger.Periodic(10),
                                       reference_positions=[(0, 0, 1),
                                                            (-1, 0, 1)])
    dim = 2 if 'polygon' in integrator.__name__.lower() else 3
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=dim,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(cl)
    sim.operations.integrator = mc

    sim.run(0)

    setattr(cl, attr, value)
    assert np.all(getattr(cl, attr) == value)


@pytest.mark.cpu
def test_pickling(simulation_factory, two_particle_snapshot_factory):
    """Test that RemoveDrift objects are picklable."""
    sim = simulation_factory(two_particle_snapshot_factory())
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1, default_a=0.1)
    mc.shape['A'] = dict(diameter=1.1)
    mc.shape['B'] = dict(diameter=1.3)
    sim.operations.integrator = mc

    cl = hoomd.hpmc.update.RemoveDrift(trigger=hoomd.trigger.Periodic(5),
                                       reference_positions=[(0, 0, 1),
                                                            (-1, 0, 1)])
    operation_pickling_check(cl, sim)
