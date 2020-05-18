# Like with the HPMC tests, include tests that set all param_dict and type_param
# entries before and after attaching.


# Test setting param dict

import hoomd
import pytest
import numpy as np
from itertools import combinations
from copy import deepcopy

np.random.seed(0)


def assert_equivalent_type_params(type_param1, type_param2):
    for pair in type_param1.to_dict():
        if isinstance(type_param1[pair], dict):
            for key in type_param1[pair]:
                assert type_param1[pair][key] == type_param2[pair][key]
        else:
            assert type_param1[pair] == type_param2[pair]


def assert_equivalent_parameter_dicts(param_dict1, param_dict2):
    for key in param_dict1:
        assert param_dict1[key] == param_dict2[key]


def _lj_params(particle_types):
    combos = list(combinations(particle_types, 2))
    sample_range = np.linspace(0.5, 1.5, len(combos) * 100)
    splt = np.array_split(np.random.choice(sample_range,
                                           size=len(combos) * 3,
                                           replace=True),
                          3)
    sigmas, epsilons, r_multipliers = splt
    return zip(combos, sigmas, epsilons, r_multipliers)


@pytest.fixture(scope="function", params=_lj_params(['A', 'B', 'C', 'D']))
def lj_params(request):
    return deepcopy(request.param)


def test_valid_params(simulation_factory, lattice_snapshot_factory, lj_params):
    combo, sigma, epsilon, r_multiplier = lj_params
    cell = hoomd.md.nlist.Cell()
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[combo] = dict(sigma=sigma, epsilon=epsilon)
    lj.r_cut[combo] = 2.5 * epsilon
    lj.r_on[combo] = lj.r_cut[combo] * r_multiplier
    assert_equivalent_type_params(lj.params, lj.params)
    assert_equivalent_type_params(lj.r_cut, lj.r_cut)
    assert_equivalent_type_params(lj.r_on, lj.r_on)
    assert_equivalent_parameter_dicts(lj.nlist._param_dict, cell._param_dict)
