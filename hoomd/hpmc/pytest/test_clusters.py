# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.update.Clusters."""

import hoomd
import pytest
import numpy as np
import hoomd.hpmc.pytest.conftest


# note: The parameterized tests validate parameters so we can't pass in values
# here that require preprocessing
valid_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10),
         delta_mu = 0,
         swap_type_pair=[],
         move_ratio=0.1,
         flip_probability=0.8,
         swap_move_ratio=0.1,
         seed=1),
    dict(trigger=hoomd.trigger.After(100),
         delta_mu=-1.5,
         swap_type_pair=['A', 'B'],
         move_ratio=0.7,
         flip_probability=1,
         swap_move_ratio=0.1,
         seed=4),
    dict(trigger=hoomd.trigger.Before(100),
         delta_mu=2.4,
         swap_type_pair=[],
         move_ratio=0.7,
         flip_probability=1,
         swap_move_ratio=0.1,
         seed=4),
    dict(trigger=hoomd.trigger.Periodic(1000),
         delta_mu=0,
         swap_type_pair=['A', 'B'],
         move_ratio=0.7,
         flip_probability=1,
         swap_move_ratio=0.1,
         seed=4),
]

valid_attrs = [
    ('trigger', hoomd.trigger.Periodic(10000)),
    ('trigger', hoomd.trigger.After(100)),
    ('trigger', hoomd.trigger.Before(12345)),
    ('swap_type_pair', []),
    ('swap_type_pair', ['A', 'B']),
    ('flip_probability', 0.2),
    ('flip_probability', 0.5),
    ('flip_probability', 0.8),
    ('move_ratio', 0.2),
    ('move_ratio', 0.5),
    ('move_ratio', 0.8),
    ('swap_move_ratio', 0.2),
    ('swap_move_ratio', 0.5),
    ('swap_move_ratio', 0.8),
    ('delta_mu', 1.8),
    ('delta_mu', -3.1),
    ('delta_mu', 2.4),
]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(constructor_args):
    """Test that Clusters can be constructed with valid arguments."""
    cl = hoomd.hpmc.update.Clusters(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(cl, attr) == value


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args,
                                       valid_args):
    """Test that Clusters can be attached with valid arguments."""

    integrator = valid_args[0]
    args = valid_args[1]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator(23456)
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"]
    mc = integrator(23456)
    mc.shape["A"] = args
    mc.shape["B"] = args

    cl = hoomd.hpmc.update.Clusters(**constructor_args)
    dim = 2 if 'polygon' in integrator.__name__.lower() else 3
    sim = simulation_factory(two_particle_snapshot_factory(particle_types=['A', 'B'],
                                                           dimensions=dim, d=2, L=40))
    sim.operations.updaters.append(cl)
    sim.operations.integrator = mc

    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(cl, attr) == value


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(attr, value):
    """Test that Clusters can get and set attributes."""
    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(10),
                                    swap_type_pair=['A', 'B'],
                                    seed=1)

    setattr(cl, attr, value)
    assert getattr(cl, attr) == value


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(attr, value, simulation_factory,
                                two_particle_snapshot_factory,
                                valid_args):
    """Test that Clusters can get and set attributes while attached."""

    integrator = valid_args[0]
    args = valid_args[1]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator(23456)
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"]
    mc = integrator(23456)
    mc.shape["A"] = args
    mc.shape["B"] = args

    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(10),
                                    swap_type_pair=['A', 'B'], seed=1)
    dim = 2 if 'polygon' in integrator.__name__.lower() else 3
    sim = simulation_factory(two_particle_snapshot_factory(particle_types=['A', 'B'],
                                                           dimensions=dim, d=2, L=40))
    sim.operations.updaters.append(cl)
    sim.operations.integrator = mc

    sim.run(0)

    setattr(cl, attr, value)
    assert getattr(cl, attr) == value


@pytest.mark.parametrize("delta_mu", [-2.0, 0, 2.0])
def test_swap_moves(delta_mu, simulation_factory,
                    lattice_snapshot_factory):
    """Test that Clusters can swap types."""

    sim = simulation_factory(lattice_snapshot_factory(particle_types=['A', 'B'],
                                                      dimensions=3, a=4, n=7, r=0.1))


    mc = hoomd.hpmc.integrate.Sphere(seed=1, d=0.1, a=0.1)
    mc.shape['A'] = dict(diameter=1)
    mc.shape['B'] = dict(diameter=1)
    sim.operations.integrator = mc

    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(5),
                                    swap_type_pair=['A', 'B'],
                                    swap_move_ratio=1,
                                    delta_mu=delta_mu,
                                    seed=12)
    sim.operations.updaters.append(cl)

    # set every other particle to type B (typeid=1)
    snap = sim.state.snapshot
    if snap.exists:
        snap.particles.typeid[range(0, sim.state.N_particles, 2)] = 1
    sim.state.snapshot = snap

    # number of particles of each type should change after a run
    sim.run(100)

    snap = sim.state.snapshot
    if snap.exists:
        num_type_B = np.sum(snap.particles.typeid)
        num_type_A = len(snap.particles.typeid) - num_type_B
        acceptance = cl.swap_moves[0]/np.sum(cl.swap_moves)
        if delta_mu<0:
            assert num_type_B < num_type_A
            assert acceptance < 1
        elif delta_mu == 0.0:
            # ALL swap moves should be accepted when delta_mu = 0
            assert np.isclose(acceptance, 1.0)
        elif delta_mu>0:
            assert num_type_A < num_type_B
            assert acceptance < 1


@pytest.mark.parametrize("delta_mu", [-2.0, 0, 2.0])
def test_pivot_moves(delta_mu, simulation_factory,
                     lattice_snapshot_factory):
    """Test that Clusters always accept pivot moves."""

    sim = simulation_factory(lattice_snapshot_factory(particle_types=['A', 'B'],
                                                      dimensions=3, a=4, n=7, r=0.1))

    mc = hoomd.hpmc.integrate.Sphere(seed=1, d=0.1, a=0.1)
    mc.shape['A'] = dict(diameter=1.1)
    mc.shape['B'] = dict(diameter=1.3)
    sim.operations.integrator = mc

    cl = hoomd.hpmc.update.Clusters(trigger=hoomd.trigger.Periodic(5),
                                    swap_type_pair=[],
                                    move_ratio=0.5,
                                    delta_mu=delta_mu,
                                    seed=12)
    sim.operations.updaters.append(cl)

    sim.run(100)

    acceptance = cl.pivot_moves[0]/np.sum(cl.pivot_moves)
    assert np.isclose(acceptance, 1.0)
