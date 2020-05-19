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
    samples = np.array_split(np.random.choice(sample_range,
                                              size=len(combos) * 3,
                                              replace=True),
                             3)
    pair_potential = [hoomd.md.pair.LJ] * len(combos)
    pair_potential_dicts = []
    for i in range(len(combos)):
        pair_potential_dicts.append({'sigma': samples[0][i],
                                    'epsilon': samples[1][i]})

    modes = np.random.choice(['none', 'shifted', 'xplor'], size=len(combos))
    return zip(pair_potential,
               pair_potential_dicts,
               combos,
               2.5 * samples[0],
               2.5 * samples[0] * samples[2],
               modes)


@pytest.fixture(scope="function", params=_lj_params(['A', 'B', 'C', 'D']))
def valid_params(request):
    return deepcopy(request.param)


def test_valid_params(valid_params):
    pair_potential, pair_potential_dict, combo, r_cut, r_on, mode = valid_params
    cell = hoomd.md.nlist.Cell()
    pot = pair_potential(nlist=cell, mode=mode)
    pot.params[combo] = pair_potential_dict
    pot.r_cut[combo] = r_cut
    pot.r_on[combo] = r_on
    assert_equivalent_type_params(pot.params, pot.params)
    assert_equivalent_type_params(pot.r_cut, pot.r_cut)
    assert_equivalent_type_params(pot.r_on, pot.r_on)
    assert_equivalent_parameter_dicts(pot.nlist._param_dict, cell._param_dict)
