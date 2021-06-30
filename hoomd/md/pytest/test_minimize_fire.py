import pytest

from hoomd import md


def _assert_correct_params(fire, param_dict):
    for param in param_dict:
        assert getattr(fire, param) == param_dict[param]


def test_get_set_params():
    fire = md.minimize.FIRE(dt=0.01)
    default_params = {'dt': 0.01,
                      'aniso': 'auto',
                      'min_steps_adapt': 5,
                      'finc_dt': 1.1,
                      'fdec_dt': 0.5,
                      'alpha_start': 0.1,
                      'fdec_alpha': 0.99,
                      'force_tol': 0.1,
                      'angmom_tol': 0.1,
                      'energy_tol': 1e-5,
                      'min_steps_conv': 10}
    _assert_correct_params(fire, default_params)

