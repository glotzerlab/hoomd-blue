# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.pair.Zetterling and HPMC pair infrastructure."""

import hoomd
import pytest
import numpy as np

valid_constructor_args = [
    {},
    dict(default_r_cut=3.0),
    dict(default_r_on=2.0),
    dict(mode="shift"),
]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that Zetterling can be constructed with valid arguments."""
    hoomd.hpmc.pair.Zetterling(**constructor_args)


@pytest.fixture(scope="session")
def mc_simulation_factory(simulation_factory, two_particle_snapshot_factory):
    """Make a MC simulation with two particles separate dy by a distance d."""

    def make_simulation(d=1):
        snapshot = two_particle_snapshot_factory(d=d)
        simulation = simulation_factory(snapshot)

        sphere = hoomd.hpmc.integrate.Sphere()
        sphere.shape["A"] = dict(diameter=0)
        simulation.operations.integrator = sphere

        return simulation

    return make_simulation


@pytest.mark.cpu
def test_attaching(mc_simulation_factory):
    """Test that Zetterling attaches."""
    zetterling = hoomd.hpmc.pair.Zetterling()
    zetterling.params[("A", "A")] = dict(
        A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649
    )

    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [zetterling]
    simulation.run(0)

    assert simulation.operations.integrator._attached
    assert zetterling._attached

    simulation.operations.integrator.pair_potentials.remove(zetterling)
    assert not zetterling._attached


invalid_parameters = [
    {},
    dict(A=1.58),
    dict(A=1.58, alpha=-0.22),
    dict(A=1.58, alpha=-0.22, kf=4.12),
    dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533),
    dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0),
    dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0),
    dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut="invalid"),
    dict(
        A=1.58,
        alpha=-0.22,
        kf=4.12,
        B=0.95533,
        sigma=1.0,
        n=18.0,
        r_cut=2.649,
        r_on="invalid",
    ),
    dict(
        A=1.58,
        alpha=-0.22,
        kf=4.12,
        B=0.95533,
        sigma=1.0,
        n=18.0,
        r_cut=2.649,
        r_on=1.0,
        invalid=10,
    ),
]


@pytest.mark.parametrize("parameters", invalid_parameters)
@pytest.mark.cpu
def test_invalid_params_on_attach(mc_simulation_factory, parameters):
    """Test that Zetterling validates parameters."""
    zetterling = hoomd.hpmc.pair.Zetterling()
    zetterling.params[("A", "A")] = dict(
        A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649
    )

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory()
    simulation.operations.integrator.pair_potentials = [zetterling]
    simulation.run(0)

    with pytest.raises(
        (
            RuntimeError,
            hoomd.error.TypeConversionError,
            KeyError,
        )
    ):
        zetterling.params[("A", "A")] = parameters


def xplor_factor(r, r_on, r_cut):
    """Compute the XPLOR smoothing factor."""
    if r < r_on:
        return 1
    if r < r_cut:
        denominator = (r_cut**2 - r_on**2) ** 3
        numerator = (r_cut**2 - r**2) ** 2 * (r_cut**2 + 2 * r**2 - 3 * r_on**2)
        return numerator / denominator

    return 0


def vzetterling(r, A, alpha, kf, B, sigma, n):
    """Compute Zetterling energy."""
    return A * np.exp(alpha * r) / r**3 * np.cos(2 * kf * r) + B * (sigma / r) ** n


# (pair params,
#  distance between particles,
#  expected energy)
zetterling_test_parameters = [
    (
        dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649),
        "none",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0),
    ),
    (
        dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649),
        "none",
        1.5,
        vzetterling(r=1.5, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0),
    ),
    (
        dict(A=1.58, alpha=-0.5, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649),
        "none",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.5, kf=4.12, B=0.95533, sigma=1.0, n=18.0),
    ),
    (
        dict(A=1.0, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649),
        "none",
        1.0,
        vzetterling(r=1.0, A=1.0, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0),
    ),
    (
        dict(A=1.58, alpha=-0.22, kf=3.5, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649),
        "none",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=3.5, B=0.95533, sigma=1.0, n=18.0),
    ),
    (
        dict(A=1.58, alpha=-0.22, kf=4.12, B=2, sigma=1.0, n=18.0, r_cut=2.649),
        "none",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=4.12, B=2, sigma=1.0, n=18.0),
    ),
    (
        dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=2.0, n=18.0, r_cut=2.649),
        "none",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=2.0, n=18.0),
    ),
    (
        dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=15.0, r_cut=2.649),
        "none",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=15.0),
    ),
    (
        dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649),
        "shift",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0)
        - vzetterling(
            r=2.649, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0
        ),
    ),
    (
        dict(A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649),
        "shift",
        3.0,
        0.0,
    ),
    (
        dict(
            A=1.58,
            alpha=-0.22,
            kf=4.12,
            B=0.95533,
            sigma=1.0,
            n=18.0,
            r_cut=2.649,
            r_on=0.8,
        ),
        "xplor",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0)
        * xplor_factor(1.0, 0.8, 2.649),
    ),
    (
        dict(
            A=1.58,
            alpha=-0.22,
            kf=4.12,
            B=0.95533,
            sigma=1.0,
            n=18.0,
            r_cut=2.649,
            r_on=0.8,
        ),
        "xplor",
        1.5,
        vzetterling(r=1.5, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0)
        * xplor_factor(1.5, 0.8, 2.649),
    ),
    (
        dict(
            A=1.58,
            alpha=-0.22,
            kf=4.12,
            B=0.95533,
            sigma=1.0,
            n=18.0,
            r_cut=2.649,
            r_on=3.0,
        ),
        "xplor",
        1.0,
        vzetterling(r=1.0, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0)
        - vzetterling(
            r=2.649, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0
        ),
    ),
    (
        dict(
            A=1.58,
            alpha=-0.22,
            kf=4.12,
            B=0.95533,
            sigma=1.0,
            n=18.0,
            r_cut=2.649,
            r_on=3.0,
        ),
        "xplor",
        2.8,
        0,
    ),
]


@pytest.mark.parametrize("params, mode, d, expected_energy", zetterling_test_parameters)
@pytest.mark.cpu
def test_energy(mc_simulation_factory, params, mode, d, expected_energy):
    """Test that Zetterling computes the correct energies for 1 pair."""
    zetterling = hoomd.hpmc.pair.Zetterling(mode=mode)
    zetterling.params[("A", "A")] = params

    simulation = mc_simulation_factory(d=d)
    simulation.operations.integrator.pair_potentials = [zetterling]
    simulation.run(0)

    assert zetterling.energy == pytest.approx(expected=expected_energy, rel=1e-5)


@pytest.mark.cpu
def test_multiple_pair_potentials(mc_simulation_factory):
    """Test that energy operates correctly with multiple pair potentials."""
    zetterling_1 = hoomd.hpmc.pair.Zetterling()
    zetterling_1.params[("A", "A")] = dict(
        A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649
    )

    zetterling_2 = hoomd.hpmc.pair.Zetterling()
    zetterling_2.params[("A", "A")] = dict(
        A=1.58, alpha=-0.5, kf=4.12, B=0.95533, sigma=1.0, n=18.0, r_cut=2.649
    )

    expected_1 = vzetterling(
        1.0, A=1.58, alpha=-0.22, kf=4.12, B=0.95533, sigma=1.0, n=18.0
    )
    expected_2 = vzetterling(
        1.0, A=1.58, alpha=-0.5, kf=4.12, B=0.95533, sigma=1.0, n=18.0
    )

    # Some parameters are validated only after attaching.
    simulation = mc_simulation_factory(1.0)
    simulation.operations.integrator.pair_potentials = [zetterling_1, zetterling_2]
    simulation.run(0)

    assert zetterling_1.energy == pytest.approx(expected=expected_1, rel=1e-5)
    assert zetterling_2.energy == pytest.approx(expected=expected_2, rel=1e-5)
    assert simulation.operations.integrator.pair_energy == pytest.approx(
        expected=expected_1 + expected_2, rel=1e-5
    )


def test_logging():
    hoomd.conftest.logging_check(
        hoomd.hpmc.pair.Zetterling,
        ("hpmc", "pair"),
        {
            "energy": {
                "category": hoomd.logging.LoggerCategories.scalar,
                "default": True,
            }
        },
    )
