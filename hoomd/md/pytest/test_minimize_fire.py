# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd.logging import LoggerCategories
from hoomd.conftest import operation_pickling_check, logging_check
from hoomd import md


def _assert_correct_params(fire, param_dict):
    """Make sure the parameters in the dictionary match with fire."""
    for param in param_dict:
        assert getattr(fire, param) == param_dict[param]


def _make_random_params():
    """Get random values for the fire parameters."""
    params = {
        'dt': np.random.rand(),
        'integrate_rotational_dof': False,
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


def test_constructor_validation():
    """Make sure constructor validates arguments."""
    with pytest.raises(ValueError):
        md.minimize.FIRE(dt=0.01,
                         force_tol=1e-1,
                         angmom_tol=1e-1,
                         energy_tol=1e-5,
                         min_steps_conv=-5)
    with pytest.raises(ValueError):
        md.minimize.FIRE(dt=0.01,
                         force_tol=1e-1,
                         angmom_tol=1e-1,
                         energy_tol=1e-5,
                         min_steps_adapt=0)


def test_get_set_params(simulation_factory, two_particle_snapshot_factory):
    """Assert we can get/set params when not attached and when attached."""
    fire = md.minimize.FIRE(dt=0.01,
                            force_tol=1e-1,
                            angmom_tol=1e-1,
                            energy_tol=1e-5)
    default_params = {
        'dt': 0.01,
        'integrate_rotational_dof': False,
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
    snap = lattice_snapshot_factory(a=1.5, n=8)
    sim = simulation_factory(snap)

    lj = md.pair.LJ(default_r_cut=2.5, nlist=md.nlist.Cell(buffer=0.4))
    lj.params[('A', 'A')] = dict(sigma=1.0, epsilon=1.0)
    nve = md.methods.ConstantVolume(hoomd.filter.All())

    fire = md.minimize.FIRE(dt=0.0025,
                            force_tol=1e-1,
                            angmom_tol=1e-1,
                            energy_tol=1e-5,
                            methods=[nve],
                            forces=[lj],
                            min_steps_conv=3)

    sim.operations.integrator = fire
    assert not fire.converged

    sim.run(0)

    initial_energy = fire.energy
    steps_to_converge = 0
    while not fire.converged:
        sim.run(1)
        steps_to_converge += 1

    assert initial_energy >= fire.energy
    assert steps_to_converge >= fire.min_steps_conv

    fire.reset()


def test_pickling(lattice_snapshot_factory, simulation_factory):
    """Assert the minimizer can be pickled when attached/unattached."""
    snap = lattice_snapshot_factory(a=1.5, n=5)
    sim = simulation_factory(snap)

    nve = md.methods.ConstantVolume(hoomd.filter.All())

    fire = md.minimize.FIRE(dt=0.0025,
                            force_tol=1e-1,
                            angmom_tol=1e-1,
                            energy_tol=1e-5,
                            methods=[nve])

    operation_pickling_check(fire, sim)


def _try_add_to_fire(sim, method, should_error=False):
    """Try adding method to FIRE's method list."""
    fire = md.minimize.FIRE(dt=0.0025,
                            force_tol=1e-1,
                            angmom_tol=1e-1,
                            energy_tol=1e-5)
    sim.operations.integrator = fire
    if should_error:
        with pytest.raises(ValueError):
            fire.methods.append(method)
    else:
        fire.methods.append(method)
    sim.run(0)


def test_validate_methods(lattice_snapshot_factory, simulation_factory):
    """Make sure only certain methods can be attached to FIRE."""
    snap = lattice_snapshot_factory(a=1.5, n=5)

    surface = md.manifold.Diamond(5)
    nve = md.methods.rattle.NVE(hoomd.filter.All(), surface)
    nph = md.methods.ConstantPressure(hoomd.filter.All(),
                                      S=1,
                                      tauS=1,
                                      couple='none')
    brownian = md.methods.Brownian(hoomd.filter.All(), kT=1)
    rattle_brownian = md.methods.rattle.Brownian(hoomd.filter.All(), 1, surface)

    methods = [(nve, False), (nph, False), (brownian, True),
               (rattle_brownian, True)]
    for method, should_error in methods:
        sim = simulation_factory(snap)
        _try_add_to_fire(sim, method, should_error)


def test_logging():
    logging_check(
        hoomd.md.minimize.FIRE, ('md', 'minimize', 'fire'), {
            'converged': {
                'category': LoggerCategories.scalar,
                'default': False
            },
            'energy': {
                'category': LoggerCategories.scalar,
                'default': True
            }
        })
