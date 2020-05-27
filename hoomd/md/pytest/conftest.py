import hoomd
import pytest
import numpy as np
from copy import deepcopy
import itertools

np.random.seed(0)

_pairs = [
    ["Gauss",
     hoomd.md.pair.Gauss(hoomd.md.nlist.Cell()),
     {'epsilon': 0.05, "sigma": 0.02}],
    ["LJ",
     hoomd.md.pair.LJ(hoomd.md.nlist.Cell()),
     {"epsilon": 0.0005, "sigma": 1}],
    ["Yukawa",
     hoomd.md.pair.Yukawa(hoomd.md.nlist.Cell()),
     {"epsilon": 0.0005, "kappa": 1}],
    ["Ewald",
     hoomd.md.pair.Ewald(hoomd.md.nlist.Cell()),
     {"alpha": 0.05, "kappa": 1}],
    ["Morse",
     hoomd.md.pair.Morse(hoomd.md.nlist.Cell()),
     {"D0": 0.05, "alpha": 1, "r0": 0}],
    ["DPDConservative",
     hoomd.md.pair.DPDConservative(hoomd.md.nlist.Cell()),
     {"A": 0.05}],
    ["ForceShiftedLJ",
     hoomd.md.pair.ForceShiftedLJ(hoomd.md.nlist.Cell()),
     {"epsilon": 0.0005, "sigma": 1}],
    ["Moliere",
     hoomd.md.pair.Moliere(hoomd.md.nlist.Cell()),
     {"Zi": 15, "Zj": 12, "a0": 1, "e": .5}],
]


@pytest.fixture(scope='function', params=_pairs, ids=(lambda x: x[0]))
def pair_and_params(request):
    return deepcopy(request.param[1:])


def _make_valid_param_dicts(key_range_dict, n_dicts):
    sample_dict = {}
    for key, val_range in key_range_dict.items():
        sample_dict[key] = np.random.choice(val_range, n_dicts)
    # turn {'a': [0, 1], 'b':[2, 3]} into [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    return [dict(zip(sample_dict, val)) for val in zip(*sample_dict.values())]


def _make_valid_params(valid_param_dicts, combos, pair_potential):
    r_cut_vals = np.random.choice(np.linspace(1.5, 3.5), len(combos))
    r_on_vals = np.random.choice(np.linspace(1.5, 3.5), len(combos))
    pair_potential_dicts = [dict(zip(combos, valid_param_dicts))] * 5
    r_cut_dicts = [dict(zip(combos, r_cut_vals))] * 5
    r_on_dicts = [dict(zip(combos, r_on_vals))] * 5
    r_on_dicts[-1] = None
    modes = ['none', 'shifted', 'xplor', None, 'none']
    pair_potentials = [pair_potential] * 5
    return zip(pair_potentials,
               pair_potential_dicts,
               r_cut_dicts,
               r_on_dicts,
               modes)


def _valid_params(particle_types=['A', 'B']):
    combos = list(itertools.combinations_with_replacement(particle_types, 2))
    lj_sample_dict = {'sigma': np.linspace(0.5, 1.5),
                      'epsilon': np.linspace(0.5, 1.5)}
    valid_param_dicts = _make_valid_param_dicts(lj_sample_dict, len(combos))
    lj_params = _make_valid_params(valid_param_dicts, combos, hoomd.md.pair.LJ)
    return lj_params


@pytest.fixture(scope="function", params=_valid_params())
def valid_params(request):
    return deepcopy(request.param)


def _make_invalid_param_dict(valid_dict):
    invalid_dicts = [valid_dict] * len(valid_dict.keys()) * 2
    count = 0
    for key in valid_dict.keys():
        invalid_dicts[count][key] = [1, 2]
        invalid_dicts[count + 1][key] = 'str'
        count += 2
    return invalid_dicts


def _make_invalid_params(valid_param_dict, invalid_param_dicts, pair_potential):
    N = len(invalid_param_dicts) + 7  # +7 is r_cut, r_on, mode, and pair key
    pair_potentials = [pair_potential] * N

    params = [{('A', 'A'): valid_param_dict}] * N
    for i in range(len(invalid_param_dicts)):
        params[i][('A', 'A')] = invalid_param_dicts[i]

    r_cuts = [{('A', 'A'): 2.5}] * N
    r_cuts[len(invalid_param_dicts)][('A', 'A')] = [1, 2]
    r_cuts[len(invalid_param_dicts) + 1][('A', 'A')] = 'str'

    r_ons = [{('A', 'A'): 2.5}] * N
    r_ons[len(invalid_param_dicts) + 2][('A', 'A')] = [1, 2]
    r_ons[len(invalid_param_dicts) + 3][('A', 'A')] = 'str'

    modes = ['none'] * N
    modes[len(invalid_param_dicts) + 4] = [1, 2]
    modes[len(invalid_param_dicts) + 5] = 1

    r_cuts[len(invalid_param_dicts) + 6] = {1: 2.4}

    return zip(pair_potentials, params, r_cuts, r_ons, modes)


def _invalid_params():
    lj_valid_param_dict = {'sigma': 1.0, 'epsilon': 1.0}
    lj_invalid_param_dicts = _make_invalid_param_dict(lj_valid_param_dict)
    lj_params = _make_invalid_params(lj_valid_param_dict,
                                     lj_invalid_param_dicts,
                                     hoomd.md.pair.LJ)
    return lj_params


@pytest.fixture(scope="function", params=_invalid_params())
def invalid_params(request):
    return deepcopy(request.param)
