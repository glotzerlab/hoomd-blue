# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.Step."""

import hoomd
import pytest


def test_valid_construction(device):
    """Test that Step can be constructed with valid arguments."""
    hoomd.hpmc.pair.Step()


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
    """Test that Step attaches."""
    step = hoomd.hpmc.pair.Step()
    step.params[('A', 'A')] = dict(epsilon=[1.0], r=[1.5])

    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [step]
    simulation.run(0)

    assert simulation.operations.integrator._attached
    assert step._attached

    simulation.operations.integrator.pair_potentials.remove(step)
    assert not step._attached


invalid_parameters = [
    {},
    dict(epsilon=[1.0]),
    dict(epsilon=[1.0], r=0.5),
    dict(epsilon=[1.0], r='invalid'),
    dict(epsilon='invalid', r=[1.0]),
    dict(epsilon=[1.0, 2.0], r=[0.5]),
    dict(epsilon=[1.0], r=[0.5], invalid=10),
    dict(epsilon=[1.0, 2.0], r=[1.0, 0.5]),
    dict(epsilon=[1, 2, 3, 4, 5], r=[0.1, 0.2, 0.3, 0.4, 0.4]),
]


@pytest.mark.parametrize("parameters", invalid_parameters)
@pytest.mark.cpu
def test_invalid_params_on_attach(mc_simulation_factory, parameters):
    """Test that Step validates parameters."""
    step = hoomd.hpmc.pair.Step()
    step.params[('A', 'A')] = dict(epsilon=[1.0], r=[1.5])

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [step]
    simulation.run(0)

    with pytest.raises((
            RuntimeError,
            hoomd.error.TypeConversionError,
            KeyError,
            ValueError,
    )):
        step.params[('A', 'A')] = parameters


# (pair params,
#  distance between particles,
#  expected energy)
step_test_parameters = [(
    dict(epsilon=[-1.125], r=[0.5]),
    3.0,
    0.0,
), (
    dict(epsilon=[-1.125], r=[0.5]),
    0.5125,
    0.0,
), (
    dict(epsilon=[-1.125], r=[0.5]),
    0.5,
    0,
), (
    dict(epsilon=[-1.125], r=[0.5]),
    0.25,
    -1.125,
), (
    dict(epsilon=[-1.125], r=[0.5]),
    0.0,
    -1.125,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    2.5,
    0,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    2.4,
    3,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    1.6,
    3,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    1.5,
    3,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    1.49,
    2,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    0.6,
    2,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    0.5,
    2,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    0.49,
    1,
), (
    dict(epsilon=[1, 2, 3], r=[0.5, 1.5, 2.5]),
    0.0,
    1,
), (
    None,
    0.0,
    0.0,
)]


@pytest.mark.parametrize('params, d, expected_energy', step_test_parameters)
@pytest.mark.cpu
def test_energy(mc_simulation_factory, params, d, expected_energy):
    """Test that Step computes the correct energies for 1 pair."""
    step = hoomd.hpmc.pair.Step()
    step.params[('A', 'A')] = params

    simulation = mc_simulation_factory(d=d)
    simulation.operations.integrator.pair_potentials = [step]
    simulation.run(0)

    assert step.energy == pytest.approx(expected=expected_energy, rel=1e-5)


def test_logging():
    hoomd.conftest.logging_check(
        hoomd.hpmc.pair.Step, ('hpmc', 'pair'), {
            'energy': {
                'category': hoomd.logging.LoggerCategories.scalar,
                'default': True
            }
        })
