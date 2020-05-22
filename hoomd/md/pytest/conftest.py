import hoomd
import pytest
import numpy as np
from copy import deepcopy
import itertools

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


def _lj_valid_params():
    particle_types_list = [['A'], ['A', 'B'],
                           ['A', 'B', 'C']]
    combos = []
    for particle_types in particle_types_list:
        type_combo = list(itertools.combinations_with_replacement(particle_types,
                                                                  2))
        combos.append(type_combo)
        N = len(type_combo)
    lj_sample_dict = {'sigma': np.linspace(0.5, 1.5),
                      'epsilon': np.linspace(0.5, 1.5)}
    valid_param_dicts = _make_valid_param_dicts(lj_sample_dict, N * len(combos))
    pair_potential_dicts = []
    r_cut_dicts = []
    r_on_dicts = []
    count = 0
    for combo_list in combos:
        combo_pair_potentials = {}
        combo_rcuts = {}
        combo_rons = {}
        for combo in combo_list:
            combo_pair_potentials[tuple(combo)] = valid_param_dicts[count]
            combo_rcuts[tuple(combo)] = (np.random.random() * 2) + 1.5
            combo_rons[tuple(combo)] = (np.random.random() * 2) + 1.5
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


@pytest.fixture(scope="function", params=_lj_valid_params())
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


def _invalid_params():
    lj_param_dict = {'sigma': 1.0, 'epsilon': 1.0}
    lj_invalid_param_dict = _make_invalid_param_dict(lj_param_dict)
    N = len(lj_invalid_param_dict) + 7  # +7 is r_cut, r_on, mode, and pair key
    lj_pot = [hoomd.md.pair.LJ] * N
    lj_params = [{('A', 'A'): lj_param_dict}] * N
    for i in range(len(lj_invalid_param_dict)):
        lj_params[i][('A', 'A')] = lj_invalid_param_dict[i]

    lj_rcuts = [{('A', 'A'): 2.5}] * N
    lj_rcuts[len(lj_invalid_param_dict)][('A', 'A')] = [1, 2]
    lj_rcuts[len(lj_invalid_param_dict) + 1][('A', 'A')] = 'str'

    lj_rons = [{('A', 'A'): 2.5}] * N
    lj_rons[len(lj_invalid_param_dict) + 2][('A', 'A')] = [1, 2]
    lj_rons[len(lj_invalid_param_dict) + 3][('A', 'A')] = 'str'

    lj_modes = ['none'] * N
    lj_modes[len(lj_invalid_param_dict) + 4] = [1, 2]
    lj_modes[len(lj_invalid_param_dict) + 5] = 1

    lj_rcuts[len(lj_invalid_param_dict) + 6] = {1: 2.4}

    return zip(lj_pot, lj_params, lj_rcuts, lj_rons, lj_modes)


@pytest.fixture(scope="function", params=_invalid_params())
def invalid_params(request):
    return deepcopy(request.param)
