# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.update.Clusters."""

import hoomd
from hoomd.conftest import (operation_pickling_check, logging_check,
                            autotuned_kernel_parameter_check)
from hoomd.logging import LoggerCategories
import pytest
import hoomd.hpmc.pytest.conftest

# note: The parameterized tests validate parameters so we can't pass in values
# here that require preprocessing
valid_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10),
         pivot_move_probability=0.1,
         flip_probability=0.8),
    dict(trigger=hoomd.trigger.After(100),
         pivot_move_probability=0.7,
         flip_probability=1),
    dict(trigger=hoomd.trigger.Before(100),
         pivot_move_probability=0.7,
         flip_probability=1),
    dict(trigger=hoomd.trigger.Periodic(1000),
         pivot_move_probability=0.7,
         flip_probability=1),
]

valid_attrs = [('trigger', hoomd.trigger.Periodic(10000)),
               ('trigger', hoomd.trigger.After(100)),
               ('trigger', hoomd.trigger.Before(12345)),
               ('flip_probability', 0.2), ('flip_probability', 0.5),
               ('flip_probability', 0.8), ('pivot_move_probability', 0.2),
               ('pivot_move_probability', 0.5), ('pivot_move_probability', 0.8)]


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that Clusters can be constructed with valid arguments."""
    cl = hoomd.hpmc.update.Clusters(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(cl, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(device, simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args, valid_args):
    """Test that Clusters can be attached with valid arguments."""
    integrator = valid_args[0]
    args = valid_args[1]
    n_dimensions = valid_args[2]
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
    mc.shape["B"] = args

    cl = hoomd.hpmc.update.Clusters(**constructor_args)
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=n_dimensions,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(cl)
    sim.operations.integrator = mc

    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(cl, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(device, attr, value):
    """Test that Clusters can get and set attributes."""
    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(10))

    setattr(cl, attr, value)
    assert getattr(cl, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(device, attr, value, simulation_factory,
                                two_particle_snapshot_factory, valid_args):
    """Test that Clusters can get and set attributes while attached."""
    integrator = valid_args[0]
    args = valid_args[1]
    n_dimensions = valid_args[2]
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
    mc.shape["B"] = args

    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(10))
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=n_dimensions,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(cl)
    sim.operations.integrator = mc

    sim.run(0)

    setattr(cl, attr, value)
    assert getattr(cl, attr) == value


@pytest.mark.serial
def test_pivot_moves(device, simulation_factory, lattice_snapshot_factory):
    """Test that Clusters produces finite size clusters."""
    if (isinstance(device, hoomd.device.GPU)
            and hoomd.version.gpu_platform == 'ROCm'):
        pytest.xfail("Clusters fails on ROCm (#1605)")

    sim = simulation_factory(
        lattice_snapshot_factory(particle_types=['A', 'B'],
                                 dimensions=3,
                                 a=4,
                                 n=7,
                                 r=0.1))

    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1, default_a=0.1)
    mc.shape['A'] = dict(diameter=1.1)
    mc.shape['B'] = dict(diameter=1.3)
    sim.operations.integrator = mc

    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(5),
                                    pivot_move_probability=0.5)
    sim.operations.updaters.append(cl)

    sim.run(10)

    avg = cl.avg_cluster_size
    assert avg > 0


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    """Test that Cluster objects are picklable."""
    sim = simulation_factory(two_particle_snapshot_factory())
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1, default_a=0.1)
    mc.shape['A'] = dict(diameter=1.1)
    mc.shape['B'] = dict(diameter=1.3)
    sim.operations.integrator = mc

    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(5),
                                    pivot_move_probability=0.1)
    operation_pickling_check(cl, sim)


@pytest.mark.serial
def test_kernel_parameters(simulation_factory, two_particle_snapshot_factory):
    """Test that Cluster objects tune their kernel parameters."""
    sim = simulation_factory(two_particle_snapshot_factory())
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1, default_a=0.1)
    mc.shape['A'] = dict(diameter=1.1)
    mc.shape['B'] = dict(diameter=1.3)
    sim.operations.integrator = mc

    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(1),
                                    pivot_move_probability=0.1)
    sim.operations.updaters.append(cl)

    sim.run(0)

    autotuned_kernel_parameter_check(instance=cl, activate=lambda: sim.run(1))


def test_logging():
    logging_check(hoomd.hpmc.update.Clusters, ('hpmc', 'update'), {
        'avg_cluster_size': {
            'category': LoggerCategories.scalar,
            'default': True
        }
    })
