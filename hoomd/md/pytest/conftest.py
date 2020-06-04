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


def _valid_params(particle_types=['A', 'B']):
    valid_params_list = []
    combos = list(itertools.combinations_with_replacement(particle_types, 2))

    lj_sample_dict = {'sigma': np.linspace(0.5, 1.5),
                      'epsilon': np.linspace(0.0005, 0.0015)}
    lj_valid_param_dicts = _make_valid_param_dicts(lj_sample_dict, len(combos))
    valid_params_list.append((hoomd.md.pair.LJ,
                              dict(zip(combos, lj_valid_param_dicts))))

    gauss_sample_dict = {'epsilon': np.linspace(0.025, 0.075),
                         'sigma': np.linspace(0.01, 0.03)}
    gauss_valid_param_dicts = _make_valid_param_dicts(gauss_sample_dict,
                                                      len(combos))
    valid_params_list.append((hoomd.md.pair.Gauss,
                              dict(zip(combos, gauss_valid_param_dicts))))

    yukawa_sample_dict = {'epsilon': np.linspace(0.00025, 0.00075),
                          'kappa': np.linspace(0.5, 1.5)}
    yukawa_valid_param_dicts = _make_valid_param_dicts(yukawa_sample_dict,
                                                       len(combos))
    valid_params_list.append((hoomd.md.pair.Yukawa,
                              dict(zip(combos, yukawa_valid_param_dicts))))

    ewald_sample_dict = {"alpha": np.linspace(0.025, 0.075),
                         "kappa": np.linspace(0.5, 1.5)}
    ewald_valid_param_dicts = _make_valid_param_dicts(ewald_sample_dict,
                                                      len(combos))
    valid_params_list.append((hoomd.md.pair.Ewald,
                              dict(zip(combos, ewald_valid_param_dicts))))

    morse_sample_dict = {"D0": np.linspace(0.025, 0.075),
                         "alpha": np.linspace(0.5, 1.5),
                         "r0": np.linspace(0, 0.1)}
    morse_valid_param_dicts = _make_valid_param_dicts(morse_sample_dict,
                                                      len(combos))
    valid_params_list.append((hoomd.md.pair.Morse,
                              dict(zip(combos, morse_valid_param_dicts))))

    dpd_conservative_sample_dict = {"A": np.linspace(0.025, 0.075)}
    dpd_conservative_valid_param_dicts = _make_valid_param_dicts(dpd_conservative_sample_dict,
                                                                 len(combos))
    valid_params_list.append((hoomd.md.pair.DPDConservative,
                              dict(zip(combos,
                                       dpd_conservative_valid_param_dicts))))

    force_shifted_LJ_sample_dict = {'sigma': np.linspace(0.5, 1.5),
                                    'epsilon': np.linspace(0.0005, 0.0015)}
    force_shifted_LJ_valid_param_dicts = _make_valid_param_dicts(force_shifted_LJ_sample_dict,
                                                                 len(combos))
    valid_params_list.append((hoomd.md.pair.ForceShiftedLJ,
                              dict(zip(combos,
                                       force_shifted_LJ_valid_param_dicts))))

    moliere_sample_dict = {'Zi': range(10, 20), 'Zj': range(8, 16),
                           'a0': np.linspace(0.5, 1.5),
                           'e': np.linspace(0.25, 0.75)}
    moliere_valid_param_dicts = _make_valid_param_dicts(moliere_sample_dict,
                                                        len(combos))
    valid_params_list.append((hoomd.md.pair.Moliere,
                              dict(zip(combos, moliere_valid_param_dicts))))
    return valid_params_list


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
    N = len(invalid_param_dicts)
    pair_potentials = [pair_potential] * N

    params = [{('A', 'A'): valid_param_dict}] * N
    for i in range(len(invalid_param_dicts)):
        params[i][('A', 'A')] = invalid_param_dicts[i]

    return zip(pair_potentials, params)


def _invalid_params():
    invalid_params_list = []

    lj_valid_dict = {'sigma': 1.0, 'epsilon': 1.0}
    lj_invalid_dicts = _make_invalid_param_dict(lj_valid_dict)
    invalid_params_list.append(_make_invalid_params(lj_valid_dict,
                                                    lj_invalid_dicts,
                                                    hoomd.md.pair.LJ))

    gauss_valid_dict = {'sigma': 0.05, 'epsilon': 0.05}
    gauss_invalid_dicts = _make_invalid_param_dict(gauss_valid_dict)
    invalid_params_list.append(_make_invalid_params(gauss_valid_dict,
                                                    gauss_invalid_dicts,
                                                    hoomd.md.pair.Gauss))

    yukawa_valid_dict = {"epsilon": 0.0005, "kappa": 1}
    yukawa_invalid_dicts = _make_invalid_param_dict(yukawa_valid_dict)
    invalid_params_list.append(_make_invalid_params(yukawa_valid_dict,
                                                    yukawa_invalid_dicts,
                                                    hoomd.md.pair.Yukawa))

    ewald_valid_dict = {"alpha": 0.05, "kappa": 1}
    ewald_invalid_dicts = _make_invalid_param_dict(ewald_valid_dict)
    invalid_params_list.append(_make_invalid_params(ewald_valid_dict,
                                                    ewald_invalid_dicts,
                                                    hoomd.md.pair.Ewald))

    morse_valid_dict = {"D0": 0.05, "alpha": 1, "r0": 0}
    morse_invalid_dicts = _make_invalid_param_dict(morse_valid_dict)
    invalid_params_list.append(_make_invalid_params(morse_valid_dict,
                                                    morse_invalid_dicts,
                                                    hoomd.md.pair.Morse))

    dpd_conservative_valid_dict = {"A": 0.05}
    dpd_conservative_invalid_dicts = _make_invalid_param_dict(dpd_conservative_valid_dict)
    invalid_params_list.append(_make_invalid_params(dpd_conservative_valid_dict,
                                                    dpd_conservative_invalid_dicts,
                                                    hoomd.md.pair.DPDConservative))

    force_shifted_LJ_valid_dict = {"epsilon": 0.0005, "sigma": 1}
    force_shifted_LJ_invalid_dicts = _make_invalid_param_dict(force_shifted_LJ_valid_dict)
    invalid_params_list.append(_make_invalid_params(force_shifted_LJ_valid_dict,
                                                    force_shifted_LJ_invalid_dicts,
                                                    hoomd.md.pair.ForceShiftedLJ))

    moliere_valid_dict = {"Zi": 15, "Zj": 12, "a0": 1, "e": .5}
    moliere_invalid_dicts = _make_invalid_param_dict(moliere_valid_dict)
    invalid_params_list.append(_make_invalid_params(moliere_valid_dict,
                                                    moliere_invalid_dicts,
                                                    hoomd.md.pair.Moliere))
    return [params for param_list in invalid_params_list for params in param_list]


@pytest.fixture(scope="function", params=_invalid_params())
def invalid_params(request):
    return deepcopy(request.param)
