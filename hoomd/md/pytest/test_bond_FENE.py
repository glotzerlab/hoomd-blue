import hoomd
import pytest
import numpy as np


def get_bond_params():
    k = [30.0, 25.0, 20.0]
    r0 = [1.6, 1.7, 1.8]
    epsilon = [0.9, 1.0, 1.1]
    sigma = [1.1, 1.0, 0.9]
    return zip(k, r0, epsilon, sigma)


@pytest.mark.parametrize("bond_params_tuple", get_bond_params())
def test_before_attaching(bond_params_tuple):
    k, r0, epsilon, sigma = bond_params_tuple
    bond_params = dict(k=k, r0=r0, epsilon=epsilon, sigma=sigma)
    bond_potential = hoomd.md.bond.FENE()
    bond_potential.params['bond'] = bond_params

    for key in bond_params.keys():
        np.testing.assert_allclose(bond_potential.params['bond'][key],
                                   bond_params[key],
                                   rtol=1e-6)
