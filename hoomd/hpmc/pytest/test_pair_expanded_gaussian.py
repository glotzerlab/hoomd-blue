# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.ExpandedGaussian and HPMC pair infrastructure."""

import hoomd
import pytest

valid_constructor_args = [
    {},
    dict(default_r_cut=2.5),
    dict(default_r_on=2.0),
    dict(mode='shift'),
]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that LennardJones can be constructed with valid arguments."""
    hoomd.hpmc.pair.LennardJones(**constructor_args)


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
    """Test that ExpandedGaussian attaches."""
    expanded_gauss = hoomd.hpmc.pair.ExpandedGaussian()
    expanded_gauss.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, r_cut=2.5)

    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [expanded_gauss]
    simulation.run(0)

    assert simulation.operations.integrator._attached
    assert expanded_gauss._attached

    simulation.operations.integrator.pair_potentials.remove(expanded_gauss)
    assert not expanded_gauss._attached


invalid_parameters = [
    {},
    dict(epsilon=1.0),
    dict(epsilon=1.0, sigma=1.0),
    dict(epsilon=1.0, sigma=1.0, r_cut='invalid'),
    dict(epsilon=1.0, sigma=1.0, r_cut=2.5, r_on='invalid'),
    dict(epsilon=1.0, sigma=1.0, r_cut=2.5, r_on=2.0, invalid=10),
]


@pytest.mark.parametrize("parameters", invalid_parameters)
@pytest.mark.cpu
def test_invalid_params_on_attach(mc_simulation_factory, parameters):
    """Test that ExpandedGaussian validates parameters."""
    expanded_gauss = hoomd.hpmc.pair.ExpandedGaussian()
    expanded_gauss.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0, r_cut=2.5)

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [expanded_gauss]
    simulation.run(0)

    with pytest.raises((
            RuntimeError,
            hoomd.error.TypeConversionError,
            KeyError,
    )):
        expanded_gauss.params[('A', 'A')] = parameters


def xplor_factor(r, r_on, r_cut):
    """Compute the XPLOR smoothing factor."""
    if r < r_on:
        return 1
    if r < r_cut:
        denominator = (r_cut**2 - r_on**2)**3
        numerator = (r_cut**2 - r**2)**2 * (r_cut**2 + 2 * r**2 - 3 * r_on**2)
        return numerator / denominator

    return 0


def lj(r, r_cut, epsilon, sigma):
    """Compute the lj energy."""
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)


# (pair params,
#  distance between particles,
#  expected energy)
expanded_gauss_test_parameters = [
    (
        dict(epsilon=2.0, sigma=1.5, r_cut=2.5),
        'none',
        3.0,
        0.0,
    ),
    (
        dict(epsilon=2.0, sigma=1.5, r_cut=2.5),
        'none',
        1.5 * 2**(1 / 6),
        -2.0,
    ),
    (
        dict(epsilon=3.0, sigma=0.5, r_cut=2.5),
        'none',
        0.5,
        0,
    ),
    (
        dict(epsilon=5.0, sigma=1.1, r_cut=2),
        'none',
        1.5,
        lj(1.5, 2, 5, 1.1),
    ),
    (
        dict(epsilon=5.0, sigma=1.1, r_cut=2),
        'shift',
        1.5,
        lj(1.5, 2, 5, 1.1) - lj(2, 2, 5, 1.1),
    ),
    (
        dict(epsilon=1.0, sigma=1, r_cut=3),
        'shift',
        3.2,
        0,
    ),
    (
        dict(epsilon=5.0, sigma=1.1, r_cut=2.5, r_on=2.0),
        'xplor',
        1.5,
        lj(1.5, 2.5, 5.0, 1.1) * xplor_factor(1.5, 2.0, 2.5),
    ),
    (
        dict(epsilon=5.0, sigma=1.1, r_cut=2.5, r_on=2.0),
        'xplor',
        2.3,
        lj(2.3, 2.5, 5, 1.1) * xplor_factor(2.3, 2.0, 2.5),
    ),
    (
        dict(epsilon=5.0, sigma=1.1, r_cut=2, r_on=3),
        'xplor',
        1.5,
        lj(1.5, 2, 5, 1.1) - lj(2, 2, 5, 1.1),
    ),
    (
        dict(epsilon=1.0, sigma=1, r_cut=3, r_on=4),
        'xplor',
        3.2,
        0,
    ),
]


@pytest.mark.parametrize('params, mode, d, expected_energy',
                         expanded_gauss_test_parameters)
@pytest.mark.cpu
def test_energy(mc_simulation_factory, params, mode, d, expected_energy):
    """Test that ExpandedGaussian computes the correct energies for 1 pair."""
    expanded_gauss = hoomd.hpmc.pair.ExpandedGaussian(mode=mode)
    expanded_gauss.params[('A', 'A')] = params

    simulation = mc_simulation_factory(d=d)
    simulation.operations.integrator.pair_potentials = [expanded_gauss]
    simulation.run(0)

    assert expanded_gauss.energy == pytest.approx(expected=expected_energy,
                                                  rel=1e-5)


@pytest.mark.cpu
def test_multiple_pair_potentials(mc_simulation_factory):
    """Test that energy operates correctly with multiple pair potentials."""
    expanded_gauss_1 = hoomd.hpmc.pair.ExpandedGaussian()
    expanded_gauss_1.params[('A', 'A')] = dict(epsilon=1.0,
                                               sigma=1.0,
                                               delta=1.0,
                                               r_cut=2.5)

    expanded_gauss_2 = hoomd.hpmc.pair.ExpandedGaussian()
    expanded_gauss_2.params[('A', 'A')] = dict(epsilon=2.0,
                                               sigma=1.0,
                                               delta=1.0,
                                               r_cut=2.5)

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory(2**(1 / 6))
    simulation.operations.integrator.pair_potentials = [
        expanded_gauss_1, expanded_gauss_2
    ]
    simulation.run(0)

    assert expanded_gauss_1.energy == pytest.approx(expected=-1.0, rel=1e-5)
    assert expanded_gauss_2.energy == pytest.approx(expected=-2.0, rel=1e-5)
    assert simulation.operations.integrator.pair_energy == pytest.approx(
        expected=-3.0, rel=1e-5)


def test_logging():
    hoomd.conftest.logging_check(
        hoomd.hpmc.pair.ExpandedGaussian, ('hpmc', 'pair'), {
            'energy': {
                'category': hoomd.logging.LoggerCategories.scalar,
                'default': True
            }
        })
