import hoomd
import pytest
import numpy as np
from itertools import combinations
from copy import deepcopy

np.random.seed(0)


def assert_equivalent_type_params(type_param1, type_param2):
    for pair in type_param1:
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
    idx = np.random.choice(len(combos), size=100, replace=True)
    combos = np.array_split([combos[i] for i in idx], 10)
    sample_range = np.linspace(0.5, 1.5, 100)
    samples = np.array_split(np.random.choice(sample_range,
                                              size=100 * 3,
                                              replace=True), 3)
    pair_potential_dicts = []
    r_cut_dicts = []
    r_on_dicts = []
    count = 0
    for combo_list in combos:
        combo_pair_potentials = {}
        combo_rcuts = {}
        combo_rons = {}
        for combo in combo_list:
            combo_pair_potentials[tuple(combo)] = {'sigma': samples[0][count],
                                                   'epsilon': samples[1][count]}
            combo_rcuts[tuple(combo)] = 2.5 * samples[0][count]
            combo_rons[tuple(combo)] = 2.5 * samples[0][count] * samples[2][count]
            count += 1
        pair_potential_dicts.append(combo_pair_potentials)
        r_cut_dicts.append(combo_rcuts)
        r_on_dicts.append(combo_rons)

    pair_potential = [hoomd.md.pair.LJ] * len(combos)
    modes = np.random.choice(['none', 'shifted', 'xplor'], size=len(combos))

    return zip(pair_potential,
               pair_potential_dicts,
               r_cut_dicts,
               r_on_dicts,
               modes)


@pytest.fixture(scope="function", params=_lj_params(['A', 'B', 'C', 'D']))
def valid_params(request):
    return deepcopy(request.param)


def test_valid_params(valid_params):
    pair_potential, pair_potential_dict, r_cut, r_on, mode = valid_params
    cell = hoomd.md.nlist.Cell()
    pot = pair_potential(nlist=cell, mode=mode)
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]
        pot.r_cut[pair] = r_cut[pair]
        pot.r_on[pair] = r_on[pair]

    assert_equivalent_type_params(pot.params.to_dict(), pair_potential_dict)
    assert_equivalent_type_params(pot.r_cut.to_dict(), r_cut)
    assert_equivalent_type_params(pot.r_on.to_dict(), r_on)
    assert_equivalent_parameter_dicts(pot.nlist._param_dict, cell._param_dict)
