# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.update.RemoveDrift."""

import hoomd
from hoomd.conftest import operation_pickling_check
import pytest
import hoomd.hpmc.pytest.conftest
import numpy as np
try:
    from mpi4py import MPI
    skip_mpi = False
except ImportError:
    skip_mpi = True

skip_mpi = pytest.mark.skipif(skip_mpi,
                              reason="MPI4py is not importable.")

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
    remove_drift = hoomd.hpmc.update.RemoveDrift(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(remove_drift, attr) == value)


@pytest.mark.cpu
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(simulation_factory,
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

    remove_drift = hoomd.hpmc.update.RemoveDrift(**constructor_args)
    dim = 2 if 'polygon' in integrator.__name__.lower() else 3
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=dim,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(remove_drift)
    sim.operations.integrator = mc

    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert np.all(getattr(remove_drift, attr) == value)


@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(device, attr, value):
    """Test that RemoveDrift can get and set attributes."""
    remove_drift = hoomd.hpmc.update.RemoveDrift(
        trigger=hoomd.trigger.Periodic(10),
        reference_positions=[(0, 0, 1), (-1, 0, 1)])

    setattr(remove_drift, attr, value)
    assert np.all(getattr(remove_drift, attr) == value)


@pytest.mark.cpu
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(attr, value, simulation_factory,
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

    remove_drift = hoomd.hpmc.update.RemoveDrift(
        trigger=hoomd.trigger.Periodic(10),
        reference_positions=[(0, 0, 1), (-1, 0, 1)])
    dim = 2 if 'polygon' in integrator.__name__.lower() else 3
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=dim,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(remove_drift)
    sim.operations.integrator = mc

    sim.run(0)

    setattr(remove_drift, attr, value)
    assert np.all(getattr(remove_drift, attr) == value)


@skip_mpi
@pytest.mark.cpu
def test_remove_drift(simulation_factory,lattice_snapshot_factory):
    """Test that RemoveDrift modifies positions correctly"""
    sim = simulation_factory(
        lattice_snapshot_factory(particle_types=['A'],
                                 dimensions=3,
                                 a=4,
                                 n=10,
                                 r=0))
    sim.seed = 19233
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1, default_a=0.1)
    mc.shape["A"] = dict(diameter=1.0)
    sim.operations.integrator = mc

    # use initial lattice configuration as reference
    comm = MPI.COMM_WORLD
    s = sim.state.snapshot
    if s.communicator.rank == 0:
        reference_positions = s.particles.position
    else:
        reference_positions = None
    reference_positions = comm.bcast(reference_positions, root=0)

    # randomize a bit
    sim.run(100)

    # remove the drift from the previous run
    remove_drift = hoomd.hpmc.update.RemoveDrift(
        trigger=hoomd.trigger.Periodic(1),
        reference_positions=reference_positions)
    sim.operations.updaters.append(remove_drift)
    sim.run(1)

    s = sim.state.snapshot
    if s.communicator.rank == 0:
        reference_com = np.mean(reference_positions,axis=0)
        new_com = np.mean(s.particles.position, axis=0)
        assert np.allclose(reference_com, new_com, atol=0.05)


@pytest.mark.cpu
def test_pickling(simulation_factory, two_particle_snapshot_factory):
    """Test that RemoveDrift objects are picklable."""
    sim = simulation_factory(two_particle_snapshot_factory())
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1, default_a=0.1)
    mc.shape['A'] = dict(diameter=1.1)
    mc.shape['B'] = dict(diameter=1.3)
    sim.operations.integrator = mc

    remove_drift = hoomd.hpmc.update.RemoveDrift(
        trigger=hoomd.trigger.Periodic(5),
        reference_positions=[(0, 0, 1), (-1, 0, 1)])
    operation_pickling_check(remove_drift, sim)
