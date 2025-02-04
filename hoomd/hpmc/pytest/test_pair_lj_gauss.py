# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.LJGauss and HPMC pair infrastructure."""

import hoomd
import pytest
import numpy as np

valid_constructor_args = [
    {},
    dict(default_r_cut=2.5),
    dict(default_r_on=2.0),
    dict(mode='shift'),
]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that LJGauss can be constructed with valid arguments."""
    hoomd.hpmc.pair.LJGauss(**constructor_args)


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
    """Test that LJGauss attaches."""
    lj_gauss = hoomd.hpmc.pair.LJGauss()
    lj_gauss.params[('A', 'A')] = dict(epsilon=1.0,
                                       sigma=0.02,
                                       r0=1.5,
                                       r_cut=2.5)

    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [lj_gauss]
    simulation.run(0)

    assert simulation.operations.integrator._attached
    assert lj_gauss._attached

    simulation.operations.integrator.pair_potentials.remove(lj_gauss)
    assert not lj_gauss._attached


invalid_parameters = [
    {},
    dict(epsilon=1.0),
    dict(epsilon=1.0, sigma=0.02),
    dict(epsilon=1.0, sigma=0.02, r0=1.5, r_cut='invalid'),
    dict(epsilon=1.0, sigma=0.02, r0=1.5, r_cut=2.5, r_on='invalid'),
    dict(epsilon=1.0, sigma=0.02, r0=1.5, r_cut=2.5, r_on=2.0, invalid=10),
]


@pytest.mark.parametrize("parameters", invalid_parameters)
@pytest.mark.cpu
def test_invalid_params_on_attach(mc_simulation_factory, parameters):
    """Test that LJGauss validates parameters."""
    lj_gauss = hoomd.hpmc.pair.LJGauss()
    lj_gauss.params[('A', 'A')] = dict(epsilon=1.0,
                                       sigma=0.02,
                                       r0=1.5,
                                       r_cut=2.5)

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [lj_gauss]
    simulation.run(0)

    with pytest.raises((
            RuntimeError,
            hoomd.error.TypeConversionError,
            KeyError,
    )):
        lj_gauss.params[('A', 'A')] = parameters


def xplor_factor(r, r_on, r_cut):
    """Compute the XPLOR smoothing factor."""
    if r < r_on:
        return 1
    if r < r_cut:
        denominator = (r_cut**2 - r_on**2)**3
        numerator = (r_cut**2 - r**2)**2 * (r_cut**2 + 2 * r**2 - 3 * r_on**2)
        return numerator / denominator

    return 0


def ljg(r, epsilon, sigma, r0):
    """Compute lj-gauss energy."""
    return (1 / r**12
            - 2 / r**6) - epsilon * np.exp(-(r - r0)**2 / 2 / sigma**2)


# (pair params,
#  distance between particles,
#  expected energy)
lj_gauss_test_parameters = [
    (
        dict(epsilon=0.0, sigma=0.02, r0=1.5, r_cut=2.5),
        'none',
        1.0,
        -1.0,
    ),
    (
        dict(epsilon=1.0, sigma=0.02, r0=1.0, r_cut=2.5),
        'none',
        1.0,
        -2.0,
    ),
    (
        dict(epsilon=1.0, sigma=0.5, r0=1.5, r_cut=2.5),
        'none',
        1.0,
        ljg(1.0, 1.0, 0.5, 1.5),
    ),
    (
        dict(epsilon=1.0, sigma=0.5, r0=1.5, r_cut=2.5),
        'none',
        1.5,
        ljg(1.5, 1.0, 0.5, 1.5),
    ),
    (
        dict(epsilon=1.0, sigma=0.5, r0=1.5, r_cut=2.5),
        'shift',
        2.0,
        ljg(2.0, 1.0, 0.5, 1.5) - ljg(2.5, 1.0, 0.5, 1.5),
    ),
    (
        dict(epsilon=1.0, sigma=0.5, r0=1.5, r_cut=2.5),
        'shift',
        2.7,
        0,
    ),
    (
        dict(epsilon=1.0, sigma=0.5, r0=1.5, r_cut=2.5, r_on=0.5),
        'xplor',
        1.0,
        ljg(1.0, 1.0, 0.5, 1.5) * xplor_factor(1.0, 0.5, 2.5),
    ),
    (
        dict(epsilon=1.0, sigma=1.0, r0=1.5, r_cut=2.5, r_on=2.0),
        'xplor',
        2.3,
        ljg(2.3, 1.0, 1.0, 1.5) * xplor_factor(2.3, 2.0, 2.5),
    ),
    (
        dict(epsilon=1.0, sigma=0.5, r0=1.5, r_cut=2.5, r_on=3.0),
        'xplor',
        1.5,
        ljg(1.5, 1.0, 0.5, 1.5) - ljg(2.5, 1.0, 0.5, 1.5),
    ),
    (
        dict(epsilon=1.0, sigma=0.5, r0=1.5, r_cut=2.5, r_on=3.0),
        'xplor',
        2.7,
        0,
    ),
]


@pytest.mark.parametrize('params, mode, d, expected_energy',
                         lj_gauss_test_parameters)
@pytest.mark.cpu
def test_energy(mc_simulation_factory, params, mode, d, expected_energy):
    """Test that LJGauss computes the correct energies for 1 pair."""
    lj_gauss = hoomd.hpmc.pair.LJGauss(mode=mode)
    lj_gauss.params[('A', 'A')] = params

    simulation = mc_simulation_factory(d=d)
    simulation.operations.integrator.pair_potentials = [lj_gauss]
    simulation.run(0)

    assert lj_gauss.energy == pytest.approx(expected=expected_energy, rel=1e-5)


@pytest.mark.cpu
def test_multiple_pair_potentials(mc_simulation_factory):
    """Test that energy operates correctly with multiple pair potentials."""
    lj_gauss_1 = hoomd.hpmc.pair.LJGauss()
    lj_gauss_1.params[('A', 'A')] = dict(epsilon=0.0,
                                         sigma=0.02,
                                         r0=1.5,
                                         r_cut=2.5)

    lj_gauss_2 = hoomd.hpmc.pair.LJGauss()
    lj_gauss_2.params[('A', 'A')] = dict(epsilon=1.0,
                                         sigma=0.02,
                                         r0=1.0,
                                         r_cut=2.5)

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory(1.0)
    simulation.operations.integrator.pair_potentials = [lj_gauss_1, lj_gauss_2]
    simulation.run(0)

    assert lj_gauss_1.energy == pytest.approx(expected=-1.0, rel=1e-5)
    assert lj_gauss_2.energy == pytest.approx(expected=-2.0, rel=1e-5)
    assert simulation.operations.integrator.pair_energy == pytest.approx(
        expected=-3.0, rel=1e-5)


def test_logging():
    hoomd.conftest.logging_check(
        hoomd.hpmc.pair.LJGauss, ('hpmc', 'pair'), {
            'energy': {
                'category': hoomd.logging.LoggerCategories.scalar,
                'default': True
            }
        })
