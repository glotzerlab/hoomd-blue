import hoomd
import pytest
import numpy as np
from copy import deepcopy
import itertools

np.random.seed(0)


def _make_valid_param_dicts(arg_dict, n_dicts):
    # turn {'a': [0, 1], 'b':[2, 3]} into [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    return [dict(zip(arg_dict, val)) for val in zip(*arg_dict.values())]


def _valid_params(particle_types=['A', 'B']):
    valid_params_list = []
    combos = list(itertools.combinations_with_replacement(particle_types, 2))

    lj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                   'epsilon': [0.0005, 0.001, 0.0015]}
    lj_valid_param_dicts = _make_valid_param_dicts(lj_arg_dict, len(combos))

    valid_params_list.append(("LJ", hoomd.md.pair.LJ, {},
                              dict(zip(combos, lj_valid_param_dicts))))

    gauss_arg_dict = {'epsilon': [0.025, 0.05, 0.075],
                      'sigma': [0.5, 1.0, 1.5]}
    gauss_valid_param_dicts = _make_valid_param_dicts(gauss_arg_dict,
                                                      len(combos))
    valid_params_list.append(("Gauss", hoomd.md.pair.Gauss, {},
                              dict(zip(combos, gauss_valid_param_dicts))))

    yukawa_arg_dict = {'epsilon': [0.00025, 0.0005, 0.00075],
                       'kappa': [0.5, 1.0, 1.5]}
    yukawa_valid_param_dicts = _make_valid_param_dicts(yukawa_arg_dict,
                                                       len(combos))
    valid_params_list.append(("Yukawa", hoomd.md.pair.Yukawa, {},
                              dict(zip(combos, yukawa_valid_param_dicts))))

    ewald_arg_dict = {"alpha": [0.025, 0.05, 0.075],
                      "kappa": [0.5, 1.0, 1.5]}
    ewald_valid_param_dicts = _make_valid_param_dicts(ewald_arg_dict,
                                                      len(combos))
    valid_params_list.append(("Ewald", hoomd.md.pair.Ewald, {},
                              dict(zip(combos, ewald_valid_param_dicts))))

    morse_arg_dict = {"D0": [0.025, 0.05, 0.075],
                      "alpha": [0.5, 1.0, 1.5],
                      "r0": [0, 0.05, 0.1]}
    morse_valid_param_dicts = _make_valid_param_dicts(morse_arg_dict,
                                                      len(combos))
    valid_params_list.append(("Morse", hoomd.md.pair.Morse, {},
                              dict(zip(combos, morse_valid_param_dicts))))

    dpd_conservative_arg_dict = {"A": [0.025, 0.05, 0.075]}
    dpd_conservative_valid_param_dicts = _make_valid_param_dicts(dpd_conservative_arg_dict,
                                                                 len(combos))
    valid_params_list.append(("DPDConservative", hoomd.md.pair.DPDConservative, {},
                              dict(zip(combos,
                                       dpd_conservative_valid_param_dicts))))

    force_shifted_LJ_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                                 'epsilon': [0.0005, 0.001, 0.0015]}
    force_shifted_LJ_valid_param_dicts = _make_valid_param_dicts(force_shifted_LJ_arg_dict,
                                                                 len(combos))
    valid_params_list.append(("ForceShiftedLJ", hoomd.md.pair.ForceShiftedLJ, {},
                              dict(zip(combos,
                                       force_shifted_LJ_valid_param_dicts))))

    moliere_arg_dict = {'Zi': [10, 15, 20], 'Zj': [8, 12, 16],
                        'a0': [0.5, 1.0, 1.5], 'e': [0.25, 0.5, 0.75]}
    moliere_valid_param_dicts = _make_valid_param_dicts(moliere_arg_dict,
                                                        len(combos))
    valid_params_list.append(("Moliere", hoomd.md.pair.Moliere, {},
                              dict(zip(combos, moliere_valid_param_dicts))))

    zbl_arg_dict = {'Zi': [10, 15, 20], 'Zj': [8, 12, 16],
                    'a0': [0.5, 1.0, 1.5], 'e': [0.25, 0.5, 0.75]}
    zbl_valid_param_dicts = _make_valid_param_dicts(zbl_arg_dict,
                                                    len(combos))
    valid_params_list.append(("ZBL", hoomd.md.pair.ZBL, {},
                              dict(zip(combos, zbl_valid_param_dicts))))

    mie_arg_dict = {'epsilon': [.05, .025, .010], 'sigma': [.5, 1, 1.5],
                    'n': [12, 14, 16], 'm': [6, 8, 10]}
    mie_valid_param_dicts = _make_valid_param_dicts(mie_arg_dict,
                                                    len(combos))
    valid_params_list.append(("Mie", hoomd.md.pair.Mie, {},
                              dict(zip(combos, mie_valid_param_dicts))))

    reactfield_arg_dict = {'epsilon': [.05, .025, .010], 'eps_rf': [.5, 1, 1.5],
                           'use_charge': [False, True, False]}
    reactfield_valid_param_dicts = _make_valid_param_dicts(reactfield_arg_dict,
                                                           len(combos))
    valid_params_list.append(("ReactionField", hoomd.md.pair.ReactionField, {},
                              dict(zip(combos, reactfield_valid_param_dicts))))

    buckingham_arg_dict = {'A': [.05, .025, .010], 'rho': [.5, 1, 1.5],
                           'C': [.05, .025, .01]}
    buckingham_valid_param_dicts = _make_valid_param_dicts(buckingham_arg_dict,
                                                           len(combos))
    valid_params_list.append(("Buckingham", hoomd.md.pair.Buckingham, {},
                              dict(zip(combos, buckingham_valid_param_dicts))))

    lj1208_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                       'epsilon': [0.0005, 0.001, 0.0015]}
    lj1208_valid_param_dicts = _make_valid_param_dicts(lj1208_arg_dict,
                                                       len(combos))

    valid_params_list.append(("LJ1208", hoomd.md.pair.LJ1208, {},
                              dict(zip(combos, lj1208_valid_param_dicts))))

    fourier_arg_dict = {'a': [[0.5, 1.0, 1.5],
                              [.05, .1, .15],
                              [.005, .01, .015]],
                        'b': [[0.25, 0.034, 0.76],
                              [0.36, 0.12, 0.65],
                              [0.78, 0.04, 0.98]]}
    fourier_valid_param_dicts = _make_valid_param_dicts(fourier_arg_dict,
                                                        len(combos))

    valid_params_list.append(("Fourier", hoomd.md.pair.Fourier, {},
                              dict(zip(combos, fourier_valid_param_dicts))))

    slj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                    'epsilon': [0.0005, 0.001, 0.0015]}
    slj_valid_param_dicts = _make_valid_param_dicts(slj_arg_dict, len(combos))

    valid_params_list.append(("SLJ", hoomd.md.pair.SLJ, {},
                              dict(zip(combos, slj_valid_param_dicts))))

    dpd_arg_dict = {'A': [0.5, 1.0, 1.5],
                    'gamma': [0.0005, 0.001, 0.0015]}
    dpd_valid_param_dicts = _make_valid_param_dicts(dpd_arg_dict, len(combos))
    dpd_extra_args = {"kT": 2}
    valid_params_list.append(("DPD", hoomd.md.pair.DPD, dpd_extra_args,
                              dict(zip(combos, dpd_valid_param_dicts))))

    dpdlj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                      'epsilon': [0.0005, 0.001, 0.0015],
                      'gamma': [0.034, 33.2, 1.2]}
    dpdlj_valid_param_dicts = _make_valid_param_dicts(dpdlj_arg_dict, len(combos))

    valid_params_list.append(("DPDLJ", hoomd.md.pair.DPDLJ, {"kT": 1},
                              dict(zip(combos, dpdlj_valid_param_dicts))))

    dlvo_arg_dict = {'kappa': [1.0, 2.0, 5.0],
                     'Z': [0.1, 0.5, 2.0],
                     'A': [0.1, 0.5, 2.0]}
    dlvo_valid_param_dicts = _make_valid_param_dicts(dlvo_arg_dict, len(combos))
    valid_params_list.append(("DLVO", hoomd.md.pair.DLVO, {},
                              dict(zip(combos, dlvo_valid_param_dicts))))
    return valid_params_list


@pytest.fixture(scope="function", params=_valid_params(), ids=(lambda x: x[0]))
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


def _make_invalid_params(valid_param_dict, invalid_param_dicts,
                         pair_potential, pair_potential_name):
    N = len(invalid_param_dicts)
    pair_potentials = [pair_potential] * N

    params = [{('A', 'A'): valid_param_dict}] * N
    for i in range(len(invalid_param_dicts)):
        params[i][('A', 'A')] = invalid_param_dicts[i]

    return zip(pair_potential_name, pair_potentials, params)


def _invalid_params():
    invalid_params_list = []

    lj_valid_dict = {'sigma': 1.0, 'epsilon': 1.0}
    lj_invalid_dicts = _make_invalid_param_dict(lj_valid_dict)
    invalid_params_list.append(_make_invalid_params(lj_valid_dict,
                                                    lj_invalid_dicts,
                                                    hoomd.md.pair.LJ, "LJ"))

    gauss_valid_dict = {'sigma': 0.05, 'epsilon': 0.05}
    gauss_invalid_dicts = _make_invalid_param_dict(gauss_valid_dict)
    invalid_params_list.append(_make_invalid_params(gauss_valid_dict,
                                                    gauss_invalid_dicts,
                                                    hoomd.md.pair.Gauss,
                                                    "Gauss"))

    yukawa_valid_dict = {"epsilon": 0.0005, "kappa": 1}
    yukawa_invalid_dicts = _make_invalid_param_dict(yukawa_valid_dict)
    invalid_params_list.append(_make_invalid_params(yukawa_valid_dict,
                                                    yukawa_invalid_dicts,
                                                    hoomd.md.pair.Yukawa,
                                                    "Yukawa"))

    ewald_valid_dict = {"alpha": 0.05, "kappa": 1}
    ewald_invalid_dicts = _make_invalid_param_dict(ewald_valid_dict)
    invalid_params_list.append(_make_invalid_params(ewald_valid_dict,
                                                    ewald_invalid_dicts,
                                                    hoomd.md.pair.Ewald,
                                                    "Ewald"))

    morse_valid_dict = {"D0": 0.05, "alpha": 1, "r0": 0}
    morse_invalid_dicts = _make_invalid_param_dict(morse_valid_dict)
    invalid_params_list.append(_make_invalid_params(morse_valid_dict,
                                                    morse_invalid_dicts,
                                                    hoomd.md.pair.Morse,
                                                    "Morse"))

    dpd_conservative_valid_dict = {"A": 0.05}
    dpd_conservative_invalid_dicts = _make_invalid_param_dict(dpd_conservative_valid_dict)
    invalid_params_list.append(_make_invalid_params(dpd_conservative_valid_dict,
                                                    dpd_conservative_invalid_dicts,
                                                    hoomd.md.pair.DPDConservative,
                                                    "DPDConservative"))

    force_shifted_LJ_valid_dict = {"epsilon": 0.0005, "sigma": 1}
    force_shifted_LJ_invalid_dicts = _make_invalid_param_dict(force_shifted_LJ_valid_dict)
    invalid_params_list.append(_make_invalid_params(force_shifted_LJ_valid_dict,
                                                    force_shifted_LJ_invalid_dicts,
                                                    hoomd.md.pair.ForceShiftedLJ,
                                                    "ForceShiftedLJ"))

    moliere_valid_dict = {"Zi": 15, "Zj": 12, "a0": 1, "e": .5}
    moliere_invalid_dicts = _make_invalid_param_dict(moliere_valid_dict)
    invalid_params_list.append(_make_invalid_params(moliere_valid_dict,
                                                    moliere_invalid_dicts,
                                                    hoomd.md.pair.Moliere,
                                                    "Moliere"))
    return [params for param_list in invalid_params_list for params in param_list]


@pytest.fixture(scope="function", params=_invalid_params(), ids=(lambda x: x[0]))
def invalid_params(request):
    return deepcopy(request.param)


def _forces_and_energies():
    params = {}
    forces = {}
    energies = {}
    extra_args = {}

    params["LJ"] = [{"sigma": 0.5, "epsilon": 0.0005},
                    {"sigma": 1.0, "epsilon": 0.001},
                    {"sigma": 1.5, "epsilon": 0.0015}]
    forces["LJ"] = [[0.00115803, 0.0000109438],
                    [-1.84064, 0.00115803],
                    [-390.144, -0.024]]
    energies["LJ"] = [[-0.000160168, -2.73972 * 10**(-6)],
                      [0.103803, -0.000320337],
                      [24.192, 0]]

    params["Gauss"] = [{"sigma": 0.5, "epsilon": 0.025},
                       {"sigma": 1.0, "epsilon": 0.05},
                       {"sigma": 1.5, "epsilon": 0.075}]
    forces["Gauss"] = [[-0.0243489, -0.00166635],
                       [-0.0283065, -0.0243489],
                       [-0.0220624, -0.0303265]]
    energies["Gauss"] = [[0.00811631, 0.000277725],
                         [0.037742, 0.0162326],
                         [0.0661873, 0.0454898]]

    params["Yukawa"] = [{"kappa": 0.5, "epsilon": 0.00025},
                        {"kappa": 1.0, "epsilon": 0.0005},
                        {"kappa": 1.5, "epsilon": 0.00075}]
    forces["Yukawa"] = [[-0.00042001, -0.0000918491],
                        [-0.000734792, -0.000123961],
                        [-0.000919849, -0.000114182]]
    energies["Yukawa"] = [[0.000229096, 0.0000787278],
                          [0.000314911, 0.0000743767],
                          [0.000324652, 0.0000526996]]

    params["Ewald"] = [{"kappa": 0.5, "alpha": 0.025},
                       {"kappa": 1.0, "alpha": 0.05},
                       {"kappa": 1.5, "alpha": 0.075}]
    forces["Ewald"] = [[-1.71273, -0.342595],
                       [-1.37038, -0.0943089],
                       [-0.834655, -0.0077883]]
    energies["Ewald"] = [[0.794344, 0.192497],
                         [0.384995, 0.0225858],
                         [0.148753, 0.000974619]]

    params["Morse"] = [{"D0": 0.025, "alpha": 0.5, "r0": 0},
                       {"D0": 0.05, "alpha": 1.0, "r0": 0.05},
                       {"D0": 0.075, "alpha": 1.5, "r0": 0.1}]
    forces["Morse"] = [[0.00537307, 0.00623091],
                       [0.0249988, 0.0179547],
                       [0.0528566, 0.0241787]]
    energies["Morse"] = [[-0.0225553, -0.0180401],
                         [-0.0373287, -0.0207059],
                         [-0.0459083, -0.0172438]]

    params["DPDConservative"] = [{"A": 0.025}, {"A": 0.05}, {"A": 0.075}]
    forces["DPDConservative"] = [[-0.0175, -0.01],
                                 [-0.035, -0.02],
                                 [-0.0525, -0.03]]
    energies["DPDConservative"] = [[0.0153125, 0.005],
                                   [0.030625, 0.01],
                                   [0.0459375, 0.015]]

    params["ForceShiftedLJ"] = [{"sigma": 0.5, "epsilon": 0.0005},
                                {"sigma": 1.0, "epsilon": 0.001},
                                {"sigma": 1.5, "epsilon": 0.0015}]
    forces["ForceShiftedLJ"] = [[0.00115772, 0.0000106367],
                                [-1.84068, 0.00111903],
                                [-390.145, -0.0246092]]
    energies["ForceShiftedLJ"] = [[-0.000159631, -2.43256 * 10**(-6)],
                                  [0.103871, -0.000281337],
                                  [24.1931, 0.000609155]]

    params["Moliere"] = [{"Zi": 10, "Zj": 8, "a0": 0.5, "e": 0.25},
                         {"Zi": 15, "Zj": 12, "a0": 1.0, "e": 0.5},
                         {"Zi": 20, "Zj": 16, "a0": 1.5, "e": 0.75}]
    forces["Moliere"] = [[-1.60329, -0.118428],
                         [-25.5994, -3.04229],
                         [-134.564, -17.5353]]
    energies["Moliere"] = [[0.440819, 0.0408002],
                           [8.75397, 1.54813],
                           [49.4355, 10.5076]]

    params["ZBL"] = [{"Zi": 10, "Zj": 8, "a0": 0.5, "e": 0.25},
                     {"Zi": 15, "Zj": 12, "a0": 1.0, "e": 0.5},
                     {"Zi": 20, "Zj": 16, "a0": 1.5, "e": 0.75}]
    forces["ZBL"] = [[-1.16329, -0.058994],
                     [-25.238, -2.20563],
                     [-141.912, -15.7028]]
    energies["ZBL"] = [[0.272618, 0.0199804],
                       [7.45095, 0.993216],
                       [47.6634, 8.11817]]

    params["Mie"] = [{"epsilon": 0.05, "sigma": 0.5, "n": 12, "m": 6},
                     {"epsilon": 0.025, "sigma": 1.0, "n": 14, "m": 8},
                     {"epsilon": 0.01, "sigma": 1.5, "n": 16, "m": 10}]
    forces["Mie"] = [[0.115803, 0.00109438],
                     [-115.77, 0.0216668],
                     [-80806.3, -0.2334687]]
    energies["Mie"] = [[-0.0160168, -0.000273972],
                       [5.67535, -0.00437856],
                       [3765.38, 0.0]]

    params["ReactionField"] = [{"epsilon": 0.05,
                                "eps_rf": 0.5,
                                "use_charge": False},
                               {"epsilon": 0.025,
                                "eps_rf": 1.0,
                                "use_charge": False},
                               {"epsilon": 0.01,
                                "eps_rf": 1.5,
                                "use_charge": False}]
    forces["ReactionField"] = [[-0.0900889, -0.0246222],
                               [-0.0444444, -0.0111111],
                               [-0.0176578, -0.00420444]]
    energies["ReactionField"] = [[0.0662167, 0.0315333],
                                 [0.0333333, 0.0166667],
                                 [0.0133783, 0.00684667]]

    params["Buckingham"] = [{"A": 0.05, "rho": 0.5, "C": 0.05},
                            {"A": 0.025, "rho": 1.0, "C": 0.025},
                            {"A": 0.01, "rho": 1.5, "C": 0.01}]
    forces["Buckingham"] = [[2.22515, 0.0125796],
                            [1.11192, 0.0032009],
                            [0.445449, 0.00105913]]
    energies["Buckingham"] = [[-0.269776, -0.00190022],
                              [-0.128657, 0.00338347],
                              [-0.0501213, 0.00280088]]

    params["LJ1208"] = [{"sigma": 0.5, "epsilon": 0.0005},
                        {"sigma": 1.0, "epsilon": 0.001},
                        {"sigma": 1.5, "epsilon": 0.0015}]
    forces["LJ1208"] = [[0.000585758, 1.59566 * 10**(-6)],
                        [-1.59425, 0.000585758],
                        [-376.832, -0.016]]
    energies["LJ1208"] = [[-0.0000626222, -3.01068 * 10**(-7)],
                          [0.0863223, -0.000125244],
                          [23.04, 0.0]]

    params["Fourier"] = [{"a": [0.5, 1.0, 1.5], "b": [0.25, 0.034, 0.76]},
                         {"a": [.05, .1, .15], "b": [0.36, 0.12, 0.65]},
                         {"a": [.005, .01, .015], "b": [0.78, 0.04, 0.98]}]
    forces["Fourier"] = [[-508.812, -5.31354],
                         [-517.515, -2.42745],
                         [-527.788, -4.27573]]
    energies["Fourier"] = [[33.0833, 1.95643],
                           [35.5141, 1.43308],
                           [39.5643, 2.47584]]

    params["SLJ"] = [{"sigma": 0.5, "epsilon": 0.0005},
                     {"sigma": 1.0, "epsilon": 0.001},
                     {"sigma": 1.5, "epsilon": 0.0015}]
    forces["SLJ"] = [[390.144, -0.024],
                     [3220830, -390.144],
                     [626907000, -76475]]
    energies["SLJ"] = [[8.064, 0.0],
                       [67092.5, 16.128],
                       [13060400, 3184.27]]

    params["DPD"] = [{"A": 0.5, "gamma": 0.0005},
                     {"A": 1.0, "gamma": 0.001},
                     {"A": 1.5, "gamma": 0.0015}]
    forces["DPD"] = [[0.0125079, 0.00714736],
                     [-0.187336, -0.107049],
                     [-0.422118, -0.24121]]
    energies["DPD"] = [[0.30625, 0.1],
                       [0.6125, 0.2],
                       [0.91875, 0.3]]

    params["DPDLJ"] = [{'sigma': 0.5, 'epsilon': 0.0005, 'gamma': 0.034},
                       {'sigma': 1.0, 'epsilon': 0.001, 'gamma': 33.2},
                       {'sigma': 1.5, 'epsilon': 0.0015, 'gamma': 1.2}]
    energies["DPDLJ"] = [[-0.000160168, -2.73972 * 10**(-6)],
                         [0.103803, -0.000320337],
                         [24.192, 0]]
    forces["DPDLJ"] = [[2.1124, 1.20774],
                       [67.8861, 37.7391],
                       [-402.7, 7.19908]]

    param_list = []
    for pair_potential in params.keys():
        kT_dict = {}
        if pair_potential == "DPD":
            kT_dict = {"kT": 2}
        elif pair_potential == "DPDLJ":
            kT_dict = {"kT": 1}
        for i in range(3):
            param_list.append((pair_potential,
                               getattr(hoomd.md.pair, pair_potential),
                               kT_dict,
                               params[pair_potential][i],
                               forces[pair_potential][i],
                               energies[pair_potential][i]))
    return param_list


@pytest.fixture(scope="function",
                params=_forces_and_energies(),
                ids=(lambda x: x[0]))
def forces_and_energies(request):
    return deepcopy(request.param)
