# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.AngularStep and HPMC pair infrastructure."""

import hoomd
import pytest

valid_constructor_args = [
    {},
    dict(delta=0.1),
    ]

@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that AngularStep can be constructed with valid arguments."""
    hoomd.hpmc.pair.AngularSteps(**constructor_args)

@pytest.fixture(scope='session')
def mc_simulation_factory(simulation_factory, two_particle_snapshot_factory):
    """Make a MC simulation with two particles separate dy by a distance d."""

    def make_simulation(d=1):
        snapshot = two_particle_snapshot_factory(d=d)
        simulation = simulation_factory(snapshot)

        sphere = hoomd.hpmc.integrate.Sphere()
        sphere.shape['A'] = dict(diameter=0)
        simulation.operations.integrator = sphere

        return simulation

    return make_simulation

@pytest.mark.cpu
def test_attaching(mc_simulation_factory):
    """Test that AngularStep attaches."""
    angular_step = hoomd.hpmc.pair.AngularStep()
    AngularStep.patch[('m')] = dict(delta=0.1)

    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [angular_step]
    simulation.run(0)

    assert simulation.operations.integrator._attached
    assert angular_step._attached

    simulation.operations.integrator.pair_potentials.remove(angular_step)
    assert not angular_step._attached

invalid_deltas = [
    {},
    dict(delta="invalid"),
]
