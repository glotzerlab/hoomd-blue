import hoomd
import pytest
import numpy as np
import itertools
from copy import deepcopy


def _make_valid_param_dicts(arg_dict):
    # turn {'a': [0, 1], 'b':[2, 3]} into [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    return [dict(zip(arg_dict, val)) for val in zip(*arg_dict.values())]


def _valid_params(particle_types=['A', 'B']):
    valid_params_list = []
    combos = list(itertools.combinations_with_replacement(particle_types, 2))

    lj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                   'epsilon': [0.0005, 0.001, 0.0015]}
    lj_valid_param_dicts = _make_valid_param_dicts(lj_arg_dict)

    valid_params_list.append((hoomd.md.pair.LJ, {},
                              dict(zip(combos, lj_valid_param_dicts))))

    gauss_arg_dict = {'epsilon': [0.025, 0.05, 0.075],
                      'sigma': [0.5, 1.0, 1.5]}
    gauss_valid_param_dicts = _make_valid_param_dicts(gauss_arg_dict)
    valid_params_list.append((hoomd.md.pair.Gauss, {},
                              dict(zip(combos, gauss_valid_param_dicts))))

    yukawa_arg_dict = {'epsilon': [0.00025, 0.0005, 0.00075],
                       'kappa': [0.5, 1.0, 1.5]}
    yukawa_valid_param_dicts = _make_valid_param_dicts(yukawa_arg_dict)
    valid_params_list.append((hoomd.md.pair.Yukawa, {},
                              dict(zip(combos, yukawa_valid_param_dicts))))

    ewald_arg_dict = {"alpha": [0.025, 0.05, 0.075],
                      "kappa": [0.5, 1.0, 1.5]}
    ewald_valid_param_dicts = _make_valid_param_dicts(ewald_arg_dict)
    valid_params_list.append((hoomd.md.pair.Ewald, {},
                              dict(zip(combos, ewald_valid_param_dicts))))

    morse_arg_dict = {"D0": [0.025, 0.05, 0.075],
                      "alpha": [0.5, 1.0, 1.5],
                      "r0": [0, 0.05, 0.1]}
    morse_valid_param_dicts = _make_valid_param_dicts(morse_arg_dict)
    valid_params_list.append((hoomd.md.pair.Morse, {},
                              dict(zip(combos, morse_valid_param_dicts))))

    dpd_conservative_arg_dict = {"A": [0.025, 0.05, 0.075]}
    dpd_conservative_valid_param_dicts = _make_valid_param_dicts(dpd_conservative_arg_dict)
    valid_params_list.append((hoomd.md.pair.DPDConservative, {},
                              dict(zip(combos,
                                       dpd_conservative_valid_param_dicts))))

    force_shifted_LJ_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                                 'epsilon': [0.0005, 0.001, 0.0015]}
    force_shifted_LJ_valid_param_dicts = _make_valid_param_dicts(force_shifted_LJ_arg_dict)
    valid_params_list.append((hoomd.md.pair.ForceShiftedLJ, {},
                              dict(zip(combos,
                                       force_shifted_LJ_valid_param_dicts))))

    moliere_arg_dict = {'qi': [2.5, 7.5, 15], 'qj': [2, 6, 12],
                        'aF': [.134197, .234463, .319536]}
    moliere_valid_param_dicts = _make_valid_param_dicts(moliere_arg_dict)
    valid_params_list.append((hoomd.md.pair.Moliere, {},
                              dict(zip(combos, moliere_valid_param_dicts))))

    zbl_arg_dict = {'qi': [2.5, 7.5, 15], 'qj': [2, 6, 12],
                    'aF': [.133669, .243535, .341914]}
    zbl_valid_param_dicts = _make_valid_param_dicts(zbl_arg_dict)
    valid_params_list.append((hoomd.md.pair.ZBL, {},
                              dict(zip(combos, zbl_valid_param_dicts))))

    mie_arg_dict = {'epsilon': [.05, .025, .010], 'sigma': [.5, 1, 1.5],
                    'n': [12, 14, 16], 'm': [6, 8, 10]}
    mie_valid_param_dicts = _make_valid_param_dicts(mie_arg_dict)
    valid_params_list.append((hoomd.md.pair.Mie, {},
                              dict(zip(combos, mie_valid_param_dicts))))

    reactfield_arg_dict = {'epsilon': [.05, .025, .010], 'eps_rf': [.5, 1, 1.5],
                           'use_charge': [False, True, False]}
    reactfield_valid_param_dicts = _make_valid_param_dicts(reactfield_arg_dict)
    valid_params_list.append((hoomd.md.pair.ReactionField, {},
                              dict(zip(combos, reactfield_valid_param_dicts))))

    buckingham_arg_dict = {'A': [.05, .025, .010], 'rho': [.5, 1, 1.5],
                           'C': [.05, .025, .01]}
    buckingham_valid_param_dicts = _make_valid_param_dicts(buckingham_arg_dict)
    valid_params_list.append((hoomd.md.pair.Buckingham, {},
                              dict(zip(combos, buckingham_valid_param_dicts))))

    lj1208_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                       'epsilon': [0.0005, 0.001, 0.0015]}
    lj1208_valid_param_dicts = _make_valid_param_dicts(lj1208_arg_dict)

    valid_params_list.append((hoomd.md.pair.LJ1208, {},
                              dict(zip(combos, lj1208_valid_param_dicts))))

    fourier_arg_dict = {'a': [[0.5, 1.0, 1.5],
                              [.05, .1, .15],
                              [.005, .01, .015]],
                        'b': [[0.25, 0.034, 0.76],
                              [0.36, 0.12, 0.65],
                              [0.78, 0.04, 0.98]]}
    fourier_valid_param_dicts = _make_valid_param_dicts(fourier_arg_dict)

    valid_params_list.append((hoomd.md.pair.Fourier, {},
                              dict(zip(combos, fourier_valid_param_dicts))))

    slj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                    'epsilon': [0.0005, 0.001, 0.0015]}
    slj_valid_param_dicts = _make_valid_param_dicts(slj_arg_dict)

    valid_params_list.append((hoomd.md.pair.SLJ, {},
                              dict(zip(combos, slj_valid_param_dicts))))

    dpd_arg_dict = {'A': [0.5, 1.0, 1.5],
                    'gamma': [0.0005, 0.001, 0.0015]}
    dpd_valid_param_dicts = _make_valid_param_dicts(dpd_arg_dict)
    dpd_extra_args = {"kT": 2}
    valid_params_list.append((hoomd.md.pair.DPD, dpd_extra_args,
                              dict(zip(combos, dpd_valid_param_dicts))))

    dpdlj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                      'epsilon': [0.0005, 0.001, 0.0015],
                      'gamma': [0.034, 33.2, 1.2]}
    dpdlj_valid_param_dicts = _make_valid_param_dicts(dpdlj_arg_dict)

    valid_params_list.append((hoomd.md.pair.DPDLJ, {"kT": 1},
                              dict(zip(combos, dpdlj_valid_param_dicts))))

    dlvo_arg_dict = {'kappa': [1.0, 2.0, 5.0],
                     'Z': [0.1, 0.5, 2.0],
                     'A': [0.1, 0.5, 2.0]}
    dlvo_valid_param_dicts = _make_valid_param_dicts(dlvo_arg_dict)
    valid_params_list.append((hoomd.md.pair.DLVO, {},
                              dict(zip(combos, dlvo_valid_param_dicts))))
    return valid_params_list


@pytest.fixture(scope="function", params=_valid_params(), ids=(lambda x: x[0].__name__))
def valid_params(request):
    return deepcopy(request.param)


def _make_invalid_param_dict(valid_dict):
    invalid_dicts = [valid_dict] * len(valid_dict.keys()) * 2
    count = 0
    for key in valid_dict.keys():
        if not isinstance(invalid_dicts[count][key], list):
            invalid_dicts[count][key] = [1, 2]
            invalid_dicts[count + 1][key] = 'str'
        else:
            invalid_dicts[count][key] = 1
            invalid_dicts[count + 1][key] = False
        count += 2
    return invalid_dicts


def _make_invalid_params(invalid_param_dicts, pot, extra_args):
    N = len(invalid_param_dicts)
    params = []
    for i in range(len(invalid_param_dicts)):
        params.append({('A', 'A'): invalid_param_dicts[i]})
    return [(pot, params[i], extra_args) for i in range(N)]


def _invalid_params():
    invalid_params_list = []

    lj_valid_dict = {'sigma': 1.0, 'epsilon': 1.0}
    lj_invalid_dicts = _make_invalid_param_dict(lj_valid_dict)
    invalid_params_list.append(_make_invalid_params(lj_invalid_dicts,
                                                    hoomd.md.pair.LJ,
                                                    {}))

    gauss_valid_dict = {'sigma': 0.05, 'epsilon': 0.05}
    gauss_invalid_dicts = _make_invalid_param_dict(gauss_valid_dict)
    invalid_params_list.append(_make_invalid_params(gauss_invalid_dicts,
                                                    hoomd.md.pair.Gauss,
                                                    {}))

    yukawa_valid_dict = {"epsilon": 0.0005, "kappa": 1}
    yukawa_invalid_dicts = _make_invalid_param_dict(yukawa_valid_dict)
    invalid_params_list.append(_make_invalid_params(yukawa_invalid_dicts,
                                                    hoomd.md.pair.Yukawa,
                                                    {}))

    ewald_valid_dict = {"alpha": 0.05, "kappa": 1}
    ewald_invalid_dicts = _make_invalid_param_dict(ewald_valid_dict)
    invalid_params_list.append(_make_invalid_params(ewald_invalid_dicts,
                                                    hoomd.md.pair.Ewald,
                                                    {}))

    morse_valid_dict = {"D0": 0.05, "alpha": 1, "r0": 0}
    morse_invalid_dicts = _make_invalid_param_dict(morse_valid_dict)
    invalid_params_list.append(_make_invalid_params(morse_invalid_dicts,
                                                    hoomd.md.pair.Morse,
                                                    {}))

    dpd_conservative_valid_dict = {"A": 0.05}
    dpd_conservative_invalid_dicts = _make_invalid_param_dict(dpd_conservative_valid_dict)
    invalid_params_list.append(_make_invalid_params(dpd_conservative_invalid_dicts,
                                                    hoomd.md.pair.DPDConservative,
                                                    {}))

    force_shifted_LJ_valid_dict = {"epsilon": 0.0005, "sigma": 1}
    force_shifted_LJ_invalid_dicts = _make_invalid_param_dict(force_shifted_LJ_valid_dict)
    invalid_params_list.append(_make_invalid_params(force_shifted_LJ_invalid_dicts,
                                                    hoomd.md.pair.ForceShiftedLJ,
                                                    {}))

    moliere_valid_dict = {"qi": 15, "qj": 12, "aF": 1}
    moliere_invalid_dicts = _make_invalid_param_dict(moliere_valid_dict)
    invalid_params_list.append(_make_invalid_params(moliere_invalid_dicts,
                                                    hoomd.md.pair.Moliere,
                                                    {}))
    zbl_valid_dict = {"qi": 10, "qj": 8, "aF": 0.5}
    zbl_invalid_dicts = _make_invalid_param_dict(zbl_valid_dict)
    invalid_params_list.append(_make_invalid_params(zbl_invalid_dicts,
                                                    hoomd.md.pair.ZBL,
                                                    {}))

    mie_valid_dict = {"epsilon": 0.05, "sigma": 0.5, "n": 12, "m": 6}
    mie_invalid_dicts = _make_invalid_param_dict(mie_valid_dict)
    invalid_params_list.append(_make_invalid_params(mie_invalid_dicts,
                                                    hoomd.md.pair.Mie,
                                                    {}))

    rf_valid_dict = {"epsilon": 0.05, "eps_rf": 0.5, "use_charge": False}
    rf_invalid_dicts = _make_invalid_param_dict(rf_valid_dict)
    invalid_params_list.append(_make_invalid_params(rf_invalid_dicts,
                                                    hoomd.md.pair.ReactionField,
                                                    {}))

    buckingham_valid_dict = {"A": 0.05, "rho": 0.5, "C": 0.05}
    buckingham_invalid_dicts = _make_invalid_param_dict(buckingham_valid_dict)
    invalid_params_list.append(_make_invalid_params(buckingham_invalid_dicts,
                                                    hoomd.md.pair.Buckingham,
                                                    {}))

    lj1208_valid_dict = {"sigma": 0.5, "epsilon": 0.0005}
    lj1208_invalid_dicts = _make_invalid_param_dict(lj1208_valid_dict)
    invalid_params_list.append(_make_invalid_params(lj1208_invalid_dicts,
                                                    hoomd.md.pair.LJ1208,
                                                    {}))

    fourier_valid_dict = {"a": [0.5, 1.0, 1.5], "b": [0.25, 0.034, 0.76]}
    fourier_invalid_dicts = _make_invalid_param_dict(fourier_valid_dict)
    invalid_params_list.append(_make_invalid_params(fourier_invalid_dicts,
                                                    hoomd.md.pair.Fourier,
                                                    {}))

    slj_valid_dict = {"sigma": 0.5, "epsilon": 0.0005}
    slj_invalid_dicts = _make_invalid_param_dict(slj_valid_dict)
    invalid_params_list.append(_make_invalid_params(slj_invalid_dicts,
                                                    hoomd.md.pair.SLJ,
                                                    {}))

    dpd_valid_dict = {"A": 0.5, "gamma": 0.0005}
    dpd_invalid_dicts = _make_invalid_param_dict(dpd_valid_dict)
    invalid_params_list.append(_make_invalid_params(dpd_invalid_dicts,
                                                    hoomd.md.pair.DPD,
                                                    {'kT': 2}))

    dpdlj_valid_dict = {'sigma': 0.5, 'epsilon': 0.0005, 'gamma': 0.034}
    dpdlj_invalid_dicts = _make_invalid_param_dict(dpdlj_valid_dict)
    invalid_params_list.append(_make_invalid_params(dpdlj_invalid_dicts,
                                                    hoomd.md.pair.DPDLJ,
                                                    {'kT': 1}))

    dlvo_valid_dict = {'kappa': 1.0, 'Z': 0.1, 'A': 0.1}
    dlvo_invalid_dicts = _make_invalid_param_dict(dlvo_valid_dict)
    invalid_params_list.append(_make_invalid_params(dlvo_invalid_dicts,
                                                    hoomd.md.pair.DLVO,
                                                    {}))
    return [params for param_list in invalid_params_list for params in param_list]


@pytest.fixture(scope="function", params=_invalid_params(), ids=(lambda x: x[0].__name__))
def invalid_params(request):
    return deepcopy(request.param)


def _forces_and_energies():
    params = {}
    forces = {}
    energies = {}

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

    params["Moliere"] = [{"qi": 2.5, "qj": 2, "aF": .134197},
                         {"qi": 7.5, "qj": 6, "aF": .234463},
                         {"qi": 15, "qj": 12, "aF": .319563}]
    forces["Moliere"] = [[-1.60329, -0.118429],
                         [-25.5994, -3.04229],
                         [-134.573, -17.5369]]
    energies["Moliere"] = [[0.44082, 0.0408005],
                           [8.75396, 1.54813],
                           [49.4399, 10.509]]

    params["ZBL"] = [{"qi": 2.5, "qj": 2, "aF": .133669},
                     {"qi": 7.5, "qj": 6, "aF": .243535},
                     {"qi": 15, "qj": 12, "aF": .341914}]
    forces["ZBL"] = [[-1.16319, -0.0589855],
                     [-25.2367, -2.20544],
                     [-141.906, -15.7017]]
    energies["ZBL"] = [[0.272589, 0.0199771],
                       [7.45044, 0.993112],
                       [47.6608, 8.11749]]

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
    forces["DPD"] = [[0.01250824, 0.00714756],
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
    forces["DPDLJ"] = [[2.11492, 1.20788],
                       [64.2114, 37.7452],
                       [-377.586, 7.15179]]

    params["DLVO"] = [{'kappa': 1.0, 'Z': 0.1, 'A': 0.1},
                      {'kappa': 2.0, 'Z': 0.5, 'A': 0.5},
                      {'kappa': 5.0, 'Z': 2.0, 'A': 2.0}]
    energies["DLVO"] = [[0.00476409, 0.00226142],
                        [0.0159279, 0.00357933],
                        [0.0188554, 0.000449299]]
    forces["DLVO"] = [[-0.00456658, -0.00226058],
                      [-0.0309878, -0.00715578],
                      [-0.0922397, -0.00225115]]

    param_list = []
    for pair_potential in params.keys():
        kT_dict = {}
        if pair_potential == "DPD":
            kT_dict = {"kT": 2}
        elif pair_potential == "DPDLJ":
            kT_dict = {"kT": 1}
        for i in range(3):
            param_list.append((getattr(hoomd.md.pair, pair_potential),
                               kT_dict,
                               params[pair_potential][i],
                               forces[pair_potential][i],
                               energies[pair_potential][i]))
    return param_list


@pytest.fixture(scope="function",
                params=_forces_and_energies(),
                ids=(lambda x: x[0].__name__))
def forces_and_energies(request):
    return deepcopy(request.param)


def _assert_equivalent_type_params(type_param1, type_param2):
    for pair in type_param1:
        if isinstance(type_param1[pair], dict):
            for key in type_param1[pair]:
                np.testing.assert_allclose(type_param1[pair][key],
                                           type_param2[pair][key])
        else:
            assert type_param1[pair] == type_param2[pair]


def _assert_equivalent_parameter_dicts(param_dict1, param_dict2):
    for key in param_dict1:
        assert param_dict1[key] == param_dict2[key]


def test_rcut(simulation_factory, two_particle_snapshot_factory):
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj.r_cut[('A', 'A')] = 'str'
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj.r_cut[('A', 'A')] = [1, 2, 3]

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.5))
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator

    _assert_equivalent_type_params(lj.r_cut.to_dict(), {('A', 'A'): 2.5})
    sim.run(1)
    _assert_equivalent_type_params(lj.r_cut.to_dict(), {('A', 'A'): 2.5})


def test_invalid_mode():
    cell = hoomd.md.nlist.Cell()
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode=1)
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode='str')
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode=[1, 2, 3])


@pytest.mark.parametrize("mode", ['none', 'shifted', 'xplor'])
def test_mode(simulation_factory, two_particle_snapshot_factory, mode):
    cell = hoomd.md.nlist.Cell()
    lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode=mode)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    snap = two_particle_snapshot_factory(dimensions=3, d=.5)
    sim = simulation_factory(snap)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.run(1)


def test_ron(simulation_factory, two_particle_snapshot_factory):
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), mode='xplor', r_cut=2.5)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj.r_on[('A', 'A')] = 'str'
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj.r_on[('A', 'A')] = [1, 2, 3]

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.5))
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    assert lj.r_on.to_dict() == {}

    lj.r_on[('A', 'A')] = 1.5
    sim.operations.schedule()
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 1.5})

    lj.r_on[('A', 'A')] = 1.0
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 1.0})


def test_valid_params(valid_params):
    pair_potential, extra_args, pair_potential_dict = valid_params
    pot = pair_potential(**extra_args,
                         nlist=hoomd.md.nlist.Cell(),
                         r_cut=2.5,
                         mode='none')
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]
    _assert_equivalent_type_params(pair_potential_dict, pot.params.to_dict())


def test_invalid_params(invalid_params):
    pair_potential, pair_potential_dict, extra_args = invalid_params
    pot = pair_potential(**extra_args, nlist=hoomd.md.nlist.Cell(), mode='none')
    for pair in pair_potential_dict:
        if isinstance(pair, tuple):
            with pytest.raises(hoomd.typeconverter.TypeConversionError):
                pot.params[pair] = pair_potential_dict[pair]


def test_invalid_pair_key():
    pot = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell())
    with pytest.raises(KeyError):
        pot.r_cut[3] = 2.5
    with pytest.raises(KeyError):
        pot.r_cut[[1, 2]] = 2.5
    with pytest.raises(KeyError):
        pot.r_cut['str'] = 2.5


def test_attached_params(simulation_factory, lattice_snapshot_factory,
                         valid_params):
    pair_potential, xtra_args, pair_potential_dict = valid_params
    pair_keys = pair_potential_dict.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = pair_potential(**xtra_args, nlist=hoomd.md.nlist.Cell(),
                         r_cut=2.5, mode='none')
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=10, a=1.5, r=0.01)

    if snap.exists:
        if 'Ewald' in str(pair_potential) and snap.exists:
            snap.particles.charge[:] = 1
        elif 'SLJ' in str(pair_potential) and snap.exists:
            snap.particles.diameter[:] = 2
        snap.particles.typeid[:] = np.random.randint(0,
                                                     len(snap.particles.types),
                                                     snap.particles.N)
    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(pot)
    sim.run(10)
    attached_pot = sim.operations.integrator.forces[0]
    _assert_equivalent_type_params(attached_pot.params.to_dict(),
                                   pair_potential_dict)


def test_run(simulation_factory, lattice_snapshot_factory, valid_params):
    pair_potential, xtra_args, pair_potential_dict = valid_params
    pair_keys = pair_potential_dict.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = pair_potential(**xtra_args, nlist=hoomd.md.nlist.Cell(),
                         r_cut=2.5, mode='none')
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=7, a=1.7, r=0.01)
    if 'Ewald' in str(pair_potential) and snap.exists:
        snap.particles.charge[:] = 1
    if snap.exists:
        snap.particles.typeid[:] = np.random.randint(0,
                                                     len(snap.particles.types),
                                                     snap.particles.N)
    sim = simulation_factory(snap)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(pot)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.operations.schedule()
    for nsteps in [3, 5, 10]:
        old_snap = sim.state.snapshot
        sim.run(nsteps)
        new_snap = sim.state.snapshot
        if new_snap.exists:
            with pytest.raises(AssertionError):
                np.testing.assert_allclose(new_snap.particles.position,
                                           old_snap.particles.position)


def test_energy_shifting(simulation_factory, two_particle_snapshot_factory):

    def S_r(r, r_cut, r_on):
        if r < r_on:
            return 1
        elif r > r_cut:
            return 0
        numerator = ((r_cut**2 - r**2)**2) * (r_cut**2 + 2 * r**2 - 3 * r_on**2)
        denominator = (r_cut**2 - r_on**2)**3
        return numerator / denominator

    r_cut = 2.5
    r_on = 0.5
    r = 1.0

    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=r_cut)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}

    snap = two_particle_snapshot_factory(dimensions=3, d=.5)
    if snap.exists:
        # avoid MPI errors by shifting positions by .1
        snap.particles.position[0] = [0, 0, .1]
        snap.particles.position[1] = [0, 0, r + .1]
    sim = simulation_factory(snap)

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.operations.schedule()

    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        E_r = sum(energies)

    snap = sim.state.snapshot
    if snap.exists:
        snap.particles.position[0] = [0, 0, .1]
        snap.particles.position[1] = [0, 0, r_cut + .1]
    sim.state.snapshot = snap
    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        E_rcut = sum(energies)

    lj_shift = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(),
                                mode='shifted', r_cut=r_cut)
    lj_shift.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.operations.schedule()

    snap = sim.state.snapshot
    if snap.exists:
        snap.particles.position[0] = [0, 0, .1]
        snap.particles.position[1] = [0, 0, r + .1]
    sim.state.snapshot = snap

    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        assert sum(energies) == E_r - E_rcut

    lj_xplor = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(),
                                mode='xplor', r_cut=r_cut)
    lj_xplor.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    lj_xplor.r_on[('A', 'A')] = 0.5
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))

    sim.operations.integrator = integrator
    sim.operations.schedule()

    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        xplor_E = sum(energies)
        assert xplor_E == E_r * S_r(r, r_cut, r_on)

        lj_xplor.r_on[('A', 'A')] = 3.0
        assert sum(energies) == E_r - E_rcut


# This function calculates the forces in a two particle simulation frame by
# finding the negative derivative of energy over inter particle distance
def _calculate_force(sim):
    snap = sim.state.snapshot
    if snap.exists:
        initial_pos = snap.particles.position
        snap.particles.position[1] = initial_pos[1] * 0.99999999
    sim.state.snapshot = snap
    E0 = sim.operations.integrator.forces[0].energies
    snap = sim.state.snapshot
    if snap.exists:
        pos = snap.particles.position
        r0 = pos[0] - pos[1]
        mag_r0 = np.linalg.norm(r0)
        direction = r0 / mag_r0

        snap.particles.position[1] = initial_pos[1] * 1.00000001
    sim.state.snapshot = snap
    E1 = sim.operations.integrator.forces[0].energies
    snap = sim.state.snapshot
    if snap.exists:
        pos = snap.particles.position
        mag_r1 = np.linalg.norm(pos[0] - pos[1])

        Fa = -1 * ((E1[0] - E0[0]) / (mag_r1 - mag_r0)) * 2 * direction
        Fb = -1 * ((E1[1] - E0[1]) / (mag_r1 - mag_r0)) * 2 * direction * -1
    snap = sim.state.snapshot
    if snap.exists:
        snap.particles.position[1] = initial_pos[1]
    sim.state.snapshot = snap
    if sim.state.snapshot.exists:
        return Fa, Fb
    else:
        return 0, 0  # return dummy values if not on rank 1


def test_force_energy_relationship(simulation_factory,
                                   two_particle_snapshot_factory,
                                   valid_params):
    # don't really test DPD and DPDLJ for this test
    pot_name = valid_params[0].__name__
    if pot_name == "DPD" or pot_name == "DPDLJ":
        pytest.skip("Cannot test force energy relationship for " +
                    pot_name + " pair force")

    pair_potential, xtra_args, pair_potential_dict = valid_params
    pair_keys = pair_potential_dict.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = pair_potential(**xtra_args, nlist=hoomd.md.nlist.Cell(),
                         r_cut=2.5, mode='none')
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]

    snap = two_particle_snapshot_factory(particle_types=particle_types, d=1.5)
    if 'Ewald' in str(pair_potential) and snap.exists:
        snap.particles.charge[:] = 1
    elif 'SLJ' in str(pair_potential) and snap.exists:
        snap.particles.diameter[:] = 2
    sim = simulation_factory(snap)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(pot)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.operations.schedule()
    for pair in pair_potential_dict:
        snap = sim.state.snapshot
        if snap.exists:
            snap.particles.typeid[0] = particle_types.index(pair[0])
            snap.particles.typeid[1] = particle_types.index(pair[1])
        sim.state.snapshot = snap

        calculated_forces = _calculate_force(sim)
        forces = sim.operations.integrator.forces[0].forces
        if forces is not None:
            sim_forces = forces
            np.testing.assert_allclose(calculated_forces[0],
                                       sim_forces[0],
                                       rtol=1e-06)
            np.testing.assert_allclose(calculated_forces[1],
                                       sim_forces[1],
                                       rtol=1e-06)


def test_force_energy_accuracy(simulation_factory,
                               two_particle_snapshot_factory,
                               forces_and_energies):
    pair_pot, extra_args, params, forces, energies = forces_and_energies
    pot = pair_pot(**extra_args, nlist=hoomd.md.nlist.Cell(),
                   r_cut=2.5, mode='none')
    pot.params[('A', 'A')] = params
    snap = two_particle_snapshot_factory(particle_types=['A'], d=0.75)
    if 'Ewald' in str(pair_pot) and snap.exists:
        snap.particles.charge[:] = 1
    elif 'SLJ' in str(pair_pot) and snap.exists:
        snap.particles.diameter[:] = 2
    elif 'DLVO' in str(pair_pot) and snap.exists:
        snap.particles.diameter[0] = 0.2
        snap.particles.diameter[1] = 0.5
    sim = simulation_factory(snap)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(pot)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.operations.schedule()
    particle_distances = [0.75, 1.5]
    for i in range(len(particle_distances)):
        d = particle_distances[i]
        r = np.array([0, 0, d]) / d
        snap = sim.state.snapshot
        if snap.exists:
            snap.particles.position[0] = [0, 0, .1]
            snap.particles.position[1] = [0, 0, d + .1]
        sim.state.snapshot = snap
        sim_energies = sim.operations.integrator.forces[0].energies
        sim_forces = sim.operations.integrator.forces[0].forces
        atol = 0
        if sim_energies is not None:
            if energies[i] == 0 or sum(sim_energies) == 0:
                atol = 1e-06
            np.testing.assert_allclose(energies[i],
                                       sum(sim_energies),
                                       rtol=5e-06,
                                       atol=atol)
        if sim_forces is not None:
            if forces[i] == 0 or sum(sim_forces[0]) == 0:
                atol = 1e-06
            np.testing.assert_allclose(forces[i] * r,
                                       sim_forces[0],
                                       rtol=5e-06,
                                       atol=atol)
            np.testing.assert_allclose(forces[i] * r * -1,
                                       sim_forces[1],
                                       rtol=5e-06,
                                       atol=atol)
