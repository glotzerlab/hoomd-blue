import numpy as np
import pytest

import hoomd
from hoomd import md


def _assert_correct_params(fire, param_dict):
    """Make sure the parameters in the dictionary match with fire."""
    for param in param_dict:
        assert getattr(fire, param) == param_dict[param]


def _make_random_params():
    """Get random values for the fire parameters."""
    params = {
        'dt': np.random.rand(),
        'aniso': 'auto',
        'min_steps_adapt': np.random.randint(1, 25),
        'finc_dt': 1 + np.random.rand(),
        'fdec_dt': np.random.rand(),
        'alpha_start': np.random.rand(),
        'fdec_alpha': np.random.rand(),
        'force_tol': np.random.rand(),
        'angmom_tol': np.random.rand(),
        'energy_tol': np.random.rand(),
        'min_steps_conv': np.random.randint(1, 15)
    }
    return params


def _set_and_check_new_params(fire):
    """Set params to random values, then assert they are correct."""
    # set the parameters to random values
    new_params = _make_random_params()
    for param in new_params:
        setattr(fire, param, new_params[param])

    # make sure they were set right
    _assert_correct_params(fire, new_params)
    return new_params


def _assert_error_if_nonpositive(fire):
    """Make sure error is raised if properties set to nonpositive values."""
    negative_value = -np.random.randint(0, 26)
    with pytest.raises(ValueError):
        fire.min_steps_adapt = negative_value

    with pytest.raises(ValueError):
        fire.min_steps_conv = negative_value


def test_get_set_params(simulation_factory, two_particle_snapshot_factory):
    """Assert we can get/set params when not attached and when attached."""
    fire = md.minimize.FIRE(dt=0.01)
    default_params = {
        'dt': 0.01,
        'aniso': 'auto',
        'min_steps_adapt': 5,
        'finc_dt': 1.1,
        'fdec_dt': 0.5,
        'alpha_start': 0.1,
        'fdec_alpha': 0.99,
        'force_tol': 0.1,
        'angmom_tol': 0.1,
        'energy_tol': 1e-5,
        'min_steps_conv': 10
    }
    _assert_correct_params(fire, default_params)

    new_params = _set_and_check_new_params(fire)

    _assert_error_if_nonpositive(fire)

    # attach to simulation
    snap = two_particle_snapshot_factory(d=2.34)
    sim = simulation_factory(snap)
    sim.operations.integrator = fire
    sim.run(0)

    # make sure the params are still right after attaching
    _assert_correct_params(fire, new_params)

    _set_and_check_new_params(fire)

    _assert_error_if_nonpositive(fire)


def test_run_minimization(lattice_snapshot_factory, simulation_factory):
    """Run a short minimization simulation."""
    snap = lattice_snapshot_factory(a=0.9, n=5)
    sim = simulation_factory(snap)

    lj = md.pair.LJ(default_r_cut=2.5, nlist=md.nlist.Cell())
    lj.params[('A', 'A')] = dict(sigma=1.0, epsilon=1.0)
    nve = md.methods.NVE(hoomd.filter.All())

    fire = md.minimize.FIRE(dt=0.0025, methods=[nve], forces=[lj])
    fire.min_steps_conv = 3

    sim.operations.integrator = fire
    fire.methods.append(nve)
    sim.run(0)

    initial_energy = fire.get_energy()
    steps_to_converge = 0
    while not fire.has_converged():
        sim.run(1)
        steps_to_converge += 1

    assert initial_energy >= fire.get_energy()
    assert steps_to_converge >= fire.min_steps_conv

    fire.reset()

    # TODO maybe a pickling test
