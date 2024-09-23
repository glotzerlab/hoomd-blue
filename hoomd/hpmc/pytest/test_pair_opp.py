# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.OPP and HPMC pair infrastructure."""

import hoomd
import pytest
import numpy as np

valid_constructor_args = [
    {},
    dict(default_r_cut=3.0),
    dict(default_r_on=2.0),
    dict(mode='shift'),
]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that OPP can be constructed with valid arguments."""
    hoomd.hpmc.pair.OPP(**constructor_args)


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
    """Test that OPP attaches."""
    opp = hoomd.hpmc.pair.OPP()
    opp.params[('A', 'A')] = dict(C1=1.,
                                  C2=1.,
                                  eta1=15,
                                  eta2=3,
                                  k=1.0,
                                  phi=np.pi,
                                  r_cut=3.0)

    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [opp]
    simulation.run(0)

    assert simulation.operations.integrator._attached
    assert opp._attached

    simulation.operations.integrator.pair_potentials.remove(opp)
    assert not opp._attached


invalid_parameters = [{},
                      dict(C1=1.),
                      dict(C1=1., C2=1.),
                      dict(C1=1., C2=1., eta1=15),
                      dict(C1=1., C2=1., eta1=15, eta2=3),
                      dict(C1=1., C2=1., eta1=15, eta2=3, k=1.0),
                      dict(C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi),
                      dict(C1=1.,
                           C2=1.,
                           eta1=15,
                           eta2=3,
                           k=1.0,
                           phi=np.pi,
                           r_cut='invalid'),
                      dict(C1=1.,
                           C2=1.,
                           eta1=15,
                           eta2=3,
                           k=1.0,
                           phi=np.pi,
                           r_cut=3.0,
                           r_on='invalid'),
                      dict(C1=1.,
                           C2=1.,
                           eta1=15,
                           eta2=3,
                           k=1.0,
                           phi=np.pi,
                           r_cut=3.0,
                           r_on=2.0,
                           invalid=10)]


@pytest.mark.parametrize("parameters", invalid_parameters)
@pytest.mark.cpu
def test_invalid_params_on_attach(mc_simulation_factory, parameters):
    """Test that OPP validates parameters."""
    opp = hoomd.hpmc.pair.OPP()
    opp.params[('A', 'A')] = dict(C1=1.,
                                  C2=1.,
                                  eta1=15,
                                  eta2=3,
                                  k=1.0,
                                  phi=np.pi,
                                  r_cut=3.0)

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [opp]
    simulation.run(0)

    with pytest.raises((
            RuntimeError,
            hoomd.error.TypeConversionError,
            KeyError,
    )):
        opp.params[('A', 'A')] = parameters


def xplor_factor(r, r_on, r_cut):
    """Compute the XPLOR smoothing factor."""
    if r < r_on:
        return 1
    if r < r_cut:
        denominator = (r_cut**2 - r_on**2)**3
        numerator = (r_cut**2 - r**2)**2 * (r_cut**2 + 2 * r**2 - 3 * r_on**2)
        return numerator / denominator

    return 0


def vopp(r, C1, C2, eta1, eta2, k, phi):
    """Compute opp energy"""
    return C1 * r**(-eta1) + C2 * r**(-eta2) * np.cos(k * r - phi)


# (pair params,
#  distance between particles,
#  expected energy)
lj_gauss_test_parameters = [
    (
        dict(C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'none',
        1.0,
        vopp(r=1.0, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=1., C2=1., eta1=12, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'none',
        1.0,
        vopp(r=1.0, C1=1., C2=1., eta1=12, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=1., C2=1., eta1=15, eta2=5, k=1.0, phi=np.pi, r_cut=3.0),
        'none',
        1.0,
        vopp(r=1.0, C1=1., C2=1., eta1=15, eta2=5, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=1., C2=1., eta1=15, eta2=3, k=3.0, phi=np.pi, r_cut=3.0),
        'none',
        1.0,
        vopp(r=1.0, C1=1., C2=1., eta1=15, eta2=3, k=3.0, phi=np.pi),
    ),
    (
        dict(C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi / 4, r_cut=3.0),
        'none',
        1.0,
        vopp(r=1.0, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi / 4),
    ),
    (
        dict(C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'none',
        1.5,
        vopp(r=1.5, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=5., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'none',
        1.0,
        vopp(r=1.0, C1=5., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=5., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'none',
        1.5,
        vopp(r=1.5, C1=5., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=5., C2=2.5, eta1=15, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'none',
        1.5,
        vopp(r=1.5, C1=5., C2=2.5, eta1=15, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'shift',
        2.0,
        vopp(r=2.0, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi)
        - vopp(r=3.0, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi, r_cut=3.0),
        'shift',
        3.2,
        0,
    ),
    (
        dict(C1=1.,
             C2=1.,
             eta1=15,
             eta2=3,
             k=1.0,
             phi=np.pi,
             r_cut=3.0,
             r_on=1.0),
        'xplor',
        1.5,
        vopp(r=1.5, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi)
        * xplor_factor(1.5, 1.0, 3.0),
    ),
    (
        dict(C1=1.,
             C2=1.,
             eta1=15,
             eta2=3,
             k=1.0,
             phi=np.pi,
             r_cut=3.0,
             r_on=2.0),
        'xplor',
        2.5,
        vopp(r=2.5, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi)
        * xplor_factor(2.5, 2.0, 3.0),
    ),
    (
        dict(C1=1.,
             C2=1.,
             eta1=15,
             eta2=3,
             k=1.0,
             phi=np.pi,
             r_cut=3.0,
             r_on=3.5),
        'xplor',
        1.5,
        vopp(r=1.5, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi)
        - vopp(r=3.0, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi),
    ),
    (
        dict(C1=1.,
             C2=1.,
             eta1=15,
             eta2=3,
             k=1.0,
             phi=np.pi,
             r_cut=3.0,
             r_on=3.5),
        'xplor',
        3.2,
        0,
    ),
]


@pytest.mark.parametrize('params, mode, d, expected_energy',
                         lj_gauss_test_parameters)
@pytest.mark.cpu
def test_energy(mc_simulation_factory, params, mode, d, expected_energy):
    """Test that OPP computes the correct energies for 1 pair."""
    opp = hoomd.hpmc.pair.OPP(mode=mode)
    opp.params[('A', 'A')] = params

    simulation = mc_simulation_factory(d=d)
    simulation.operations.integrator.pair_potentials = [opp]
    simulation.run(0)

    assert opp.energy == pytest.approx(expected=expected_energy, rel=1e-5)


@pytest.mark.cpu
def test_multiple_pair_potentials(mc_simulation_factory):
    """Test that energy operates correctly with multiple pair potentials."""
    opp_1 = hoomd.hpmc.pair.OPP()
    opp_1.params[('A', 'A')] = dict(C1=1.,
                                    C2=1.,
                                    eta1=15,
                                    eta2=3,
                                    k=1.0,
                                    phi=np.pi,
                                    r_cut=3.0)

    opp_2 = hoomd.hpmc.pair.OPP()
    opp_2.params[('A', 'A')] = dict(C1=5.,
                                    C2=1.,
                                    eta1=15,
                                    eta2=3,
                                    k=1.0,
                                    phi=np.pi,
                                    r_cut=3.0)

    expected_1 = vopp(1.0, C1=1., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi)
    expected_2 = vopp(1.0, C1=5., C2=1., eta1=15, eta2=3, k=1.0, phi=np.pi)

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory(1.0)
    simulation.operations.integrator.pair_potentials = [opp_1, opp_2]
    simulation.run(0)

    assert opp_1.energy == pytest.approx(expected=expected_1, rel=1e-5)
    assert opp_2.energy == pytest.approx(expected=expected_2, rel=1e-5)
    assert simulation.operations.integrator.pair_energy == pytest.approx(
        expected=expected_1 + expected_2, rel=1e-5)


def test_logging():
    hoomd.conftest.logging_check(
        hoomd.hpmc.pair.OPP, ('hpmc', 'pair'), {
            'energy': {
                'category': hoomd.logging.LoggerCategories.scalar,
                'default': True
            }
        })
