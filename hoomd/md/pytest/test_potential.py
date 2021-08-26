from collections.abc import Sequence, Mapping
import math
from numbers import Number

import numpy as np

import hoomd
from hoomd import md
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check, pickling_check
from hoomd.error import TypeConversionError
import pytest
import itertools
from copy import deepcopy
import json
from pathlib import Path
from collections import namedtuple


def _equivalent_data_structures(reference, struct_2):
    """Compare arbitrary data structures for equality.

    ``reference`` is expected to be the reference data structure. Cannot handle
    set like data structures.
    """
    if isinstance(reference, np.ndarray):
        return np.allclose(reference, struct_2)
    if isinstance(reference, Mapping):
        # if the non-reference value does not have all the keys
        # we don't check for the exact same keys, since some values may have
        # defaults.
        if set(reference.keys()) - set(struct_2.keys()):
            return False
        return all(
            _equivalent_data_structures(reference[key], struct_2[key])
            for key in reference)
    if isinstance(reference, Sequence):
        if len(reference) != len(struct_2):
            return False
        return all(
            _equivalent_data_structures(value_1, value_2)
            for value_1, value_2 in zip(reference, struct_2))
    if isinstance(reference, Number):
        return math.isclose(reference, struct_2)


def _assert_equivalent_parameter_dicts(param_dict1, param_dict2):
    for key in param_dict1:
        assert param_dict1[key] == param_dict2[key]


def test_rcut(simulation_factory, two_particle_snapshot_factory):
    lj = md.pair.LJ(nlist=md.nlist.Cell(), default_r_cut=2.5)
    assert lj.r_cut.default == 2.5

    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    with pytest.raises(TypeConversionError):
        lj.r_cut[('A', 'A')] = 'str'
    with pytest.raises(TypeConversionError):
        lj.r_cut[('A', 'A')] = [1, 2, 3]

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.5))
    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator

    lj.r_cut[('A', 'A')] = 2.5
    assert _equivalent_data_structures({('A', 'A'): 2.5}, lj.r_cut.to_dict())
    sim.run(0)
    assert _equivalent_data_structures({('A', 'A'): 2.5}, lj.r_cut.to_dict())


def test_invalid_mode():
    cell = md.nlist.Cell()
    for invalid_mode in [1, 'str', [1, 2, 3]]:
        with pytest.raises(TypeConversionError):
            md.pair.LJ(nlist=cell, default_r_cut=2.5, mode=invalid_mode)


@pytest.mark.parametrize("mode", ['none', 'shift', 'xplor'])
def test_mode(simulation_factory, two_particle_snapshot_factory, mode):
    cell = md.nlist.Cell()
    lj = md.pair.LJ(nlist=cell, default_r_cut=2.5, mode=mode)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    snap = two_particle_snapshot_factory(dimensions=3, d=.5)
    sim = simulation_factory(snap)
    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    sim.run(1)


def test_ron(simulation_factory, two_particle_snapshot_factory):
    lj = md.pair.LJ(nlist=md.nlist.Cell(), mode='xplor', default_r_cut=2.5)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    with pytest.raises(TypeConversionError):
        lj.r_on[('A', 'A')] = 'str'
    with pytest.raises(TypeConversionError):
        lj.r_on[('A', 'A')] = [1, 2, 3]

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.5))
    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    assert lj.r_on.to_dict() == {}

    lj.r_on[('A', 'A')] = 1.5
    assert _equivalent_data_structures({('A', 'A'): 1.5}, lj.r_on.to_dict())
    sim.run(0)
    assert _equivalent_data_structures({('A', 'A'): 1.5}, lj.r_on.to_dict())

    lj.r_on[('A', 'A')] = 1.0
    assert _equivalent_data_structures({('A', 'A'): 1.0}, lj.r_on.to_dict())


def _make_invalid_param_dict(valid_dict):
    """This could is fragile if multiple types are allowed for a key."""
    invalid_dicts = [valid_dict] * len(valid_dict.keys()) * 2
    count = 0
    for key in valid_dict.keys():
        invalid_count = 0
        # Set one invalid argument per dictionary
        # Set two invalid arguments per key
        valid_value = invalid_dicts[count][key]
        if not isinstance(valid_value, (list, np.ndarray)):
            invalid_dicts[count][key] = [1, 2]
            invalid_count += 1
        if not isinstance(valid_value, (str, np.ndarray)):
            invalid_dicts[count + 1][key] = 'str'
            invalid_count += 1
        if invalid_count == 2:
            break
        if not isinstance(valid_value, float):
            invalid_dicts[count][key] = 1.0
            invalid_count += 1
        if invalid_count == 2:
            break
        if not isinstance(valid_value, bool):
            invalid_dicts[count + 1][key] = False
            invalid_count += 1
        if invalid_count != 2:
            raise RuntimeError("Unable to generate 2 invalid dict entries.")
        count += 2
    return invalid_dicts


paramtuple = namedtuple(
    'paramtuple', ['pair_potential', 'pair_potential_params', 'extra_args'])


def _make_invalid_params(invalid_param_dicts, pot, extra_args):
    N = len(invalid_param_dicts)
    params = []
    for i in range(len(invalid_param_dicts)):
        params.append({('A', 'A'): invalid_param_dicts[i]})
    return [paramtuple(pot, params[i], extra_args) for i in range(N)]


def _invalid_params():
    invalid_params_list = []
    # Start with valid parameters to get the keys and placeholder values

    lj_valid_dict = {'sigma': 1.0, 'epsilon': 1.0}
    lj_invalid_dicts = _make_invalid_param_dict(lj_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(lj_invalid_dicts, md.pair.LJ, {}))

    gauss_valid_dict = {'sigma': 0.05, 'epsilon': 0.05}
    gauss_invalid_dicts = _make_invalid_param_dict(gauss_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(gauss_invalid_dicts, md.pair.Gauss, {}))

    yukawa_valid_dict = {"epsilon": 0.0005, "kappa": 1}
    yukawa_invalid_dicts = _make_invalid_param_dict(yukawa_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(yukawa_invalid_dicts, md.pair.Yukawa, {}))

    ewald_valid_dict = {"alpha": 0.05, "kappa": 1}
    ewald_invalid_dicts = _make_invalid_param_dict(ewald_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(ewald_invalid_dicts, md.pair.Ewald, {}))

    morse_valid_dict = {"D0": 0.05, "alpha": 1, "r0": 0}
    morse_invalid_dicts = _make_invalid_param_dict(morse_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(morse_invalid_dicts, md.pair.Morse, {}))

    dpd_conservative_valid_dict = {"A": 0.05}
    dpd_conservative_invalid_dicts = _make_invalid_param_dict(
        dpd_conservative_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(dpd_conservative_invalid_dicts,
                             md.pair.DPDConservative, {}))

    force_shifted_LJ_valid_dict = {"epsilon": 0.0005, "sigma": 1}
    force_shifted_LJ_invalid_dicts = _make_invalid_param_dict(
        force_shifted_LJ_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(force_shifted_LJ_invalid_dicts,
                             md.pair.ForceShiftedLJ, {}))

    moliere_valid_dict = {"qi": 15, "qj": 12, "aF": 1}
    moliere_invalid_dicts = _make_invalid_param_dict(moliere_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(moliere_invalid_dicts, md.pair.Moliere, {}))
    zbl_valid_dict = {"qi": 10, "qj": 8, "aF": 0.5}
    zbl_invalid_dicts = _make_invalid_param_dict(zbl_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(zbl_invalid_dicts, md.pair.ZBL, {}))

    mie_valid_dict = {"epsilon": 0.05, "sigma": 0.5, "n": 12, "m": 6}
    mie_invalid_dicts = _make_invalid_param_dict(mie_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(mie_invalid_dicts, md.pair.Mie, {}))

    rf_valid_dict = {"epsilon": 0.05, "eps_rf": 0.5, "use_charge": False}
    rf_invalid_dicts = _make_invalid_param_dict(rf_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(rf_invalid_dicts, md.pair.ReactionField, {}))

    buckingham_valid_dict = {"A": 0.05, "rho": 0.5, "C": 0.05}
    buckingham_invalid_dicts = _make_invalid_param_dict(buckingham_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(buckingham_invalid_dicts, md.pair.Buckingham, {}))

    lj1208_valid_dict = {"sigma": 0.5, "epsilon": 0.0005}
    lj1208_invalid_dicts = _make_invalid_param_dict(lj1208_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(lj1208_invalid_dicts, md.pair.LJ1208, {}))

    lj0804_valid_dict = {'sigma': 1.0, 'epsilon': 1.0}
    lj0804_invalid_dicts = _make_invalid_param_dict(lj0804_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(lj0804_invalid_dicts, md.pair.LJ0804, {}))

    fourier_valid_dict = {"a": [0.5, 1.0, 1.5], "b": [0.25, 0.034, 0.76]}
    fourier_invalid_dicts = _make_invalid_param_dict(fourier_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(fourier_invalid_dicts, md.pair.Fourier, {}))

    slj_valid_dict = {"sigma": 0.5, "epsilon": 0.0005}
    slj_invalid_dicts = _make_invalid_param_dict(slj_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(slj_invalid_dicts, md.pair.SLJ, {}))

    expanded_mie_valid_dict = {
        "epsilon": 0.05,
        "sigma": 0.5,
        "n": 12,
        "m": 6,
        "delta": 0.25
    }
    expanded_mie_invalid_dicts = _make_invalid_param_dict(
        expanded_mie_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(expanded_mie_invalid_dicts, md.pair.ExpandedMie,
                             {}))

    dpd_valid_dict = {"A": 0.5, "gamma": 0.0005}
    dpd_invalid_dicts = _make_invalid_param_dict(dpd_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(dpd_invalid_dicts, md.pair.DPD, {'kT': 2}))

    dpdlj_valid_dict = {'sigma': 0.5, 'epsilon': 0.0005, 'gamma': 0.034}
    dpdlj_invalid_dicts = _make_invalid_param_dict(dpdlj_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(dpdlj_invalid_dicts, md.pair.DPDLJ, {'kT': 1}))

    dlvo_valid_dict = {'kappa': 1.0, 'Z': 0.1, 'A': 0.1}
    dlvo_invalid_dicts = _make_invalid_param_dict(dlvo_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(dlvo_invalid_dicts, md.pair.DLVO, {}))

    opp_valid_dict = {
        'C1': 1.0,
        'C2': 0.1,
        'eta1': 15,
        'eta2': 3,
        'k': 0.8,
        'phi': 0.1
    }
    opp_invalid_dicts = _make_invalid_param_dict(opp_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(opp_invalid_dicts, hoomd.md.pair.OPP, {}))

    twf_valid_dict = {'sigma': 1.0, 'epsilon': 1.0, 'alpha': 15}
    twf_invalid_dicts = _make_invalid_param_dict(twf_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(twf_invalid_dicts, hoomd.md.pair.TWF, {}))

    cossq_valid_dict = {'sigma': 1.0, 'epsilon': 1.0, 'wca': True}
    cossq_invalid_dicts = _make_invalid_param_dict(cossq_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(cossq_invalid_dicts, hoomd.md.pair.CosineSquared, 
                                {}))

    table_valid_dict = {
        'V': np.arange(0, 20, 1) / 10,
        'F': np.asarray(20 * [-1.9 / 2.5]),
        'r_min': 0.0
    }
    table_invalid_dicts = _make_invalid_param_dict(table_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(table_invalid_dicts, hoomd.md.pair.Table, {}))

    tersoff_valid_dict = {
        'cutoff_thickness': 1.0,
        'magnitudes': (5.0, 2.0),
        'exp_factors': (2.0, 2.0),
        'lambda3': 2.0,
        'dimer_r': 2.5,
        'n': 2.0,
        'gamma': 2.0,
        'c': 2.0,
        'd': 2.0,
        'm': 2.0,
        'alpha': 2.0,
    }
    tersoff_invalid_dicts = _make_invalid_param_dict(tersoff_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(tersoff_invalid_dicts, hoomd.md.many_body.Tersoff,
                             {}))

    square_density_valid_dict = {'A': 5.0, 'B': 2.0}
    sq_dens_invalid_dicts = _make_invalid_param_dict(square_density_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(sq_dens_invalid_dicts,
                             hoomd.md.many_body.SquareDensity, {}))

    revcross_valid_dict = {
        'sigma': 5.0,
        'n': 2.0,
        'epsilon': 2.0,
        'lambda3': 2.0
    }
    revcross_invalid_dicts = _make_invalid_param_dict(revcross_valid_dict)
    invalid_params_list.extend(
        _make_invalid_params(revcross_invalid_dicts,
                             hoomd.md.many_body.RevCross, {}))

    return invalid_params_list


@pytest.fixture(scope="function",
                params=_invalid_params(),
                ids=(lambda x: x[0].__name__))
def invalid_params(request):
    return deepcopy(request.param)


def test_invalid_params(invalid_params):
    pot = invalid_params.pair_potential(**invalid_params.extra_args,
                                        nlist=md.nlist.Cell())
    for pair in invalid_params.pair_potential_params:
        if isinstance(pair, tuple):
            with pytest.raises(TypeConversionError):
                pot.params[pair] = invalid_params.pair_potential_params[pair]


def test_invalid_pair_key():
    pot = md.pair.LJ(nlist=md.nlist.Cell())
    for invalid_key in [3, [1, 2], 'str']:
        with pytest.raises(KeyError):
            pot.r_cut[invalid_key] = 2.5


def _make_valid_param_dicts(arg_dict):
    """Unpack dictionary of lists of numbers into dictionary of numbers.

    Ex: turn {'a': [0, 1], 'b':[2, 3]} into [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    """
    return [dict(zip(arg_dict, val)) for val in zip(*arg_dict.values())]


def _valid_params(particle_types=['A', 'B']):
    valid_params_list = []
    combos = list(itertools.combinations_with_replacement(particle_types, 2))
    lj_arg_dict = {'sigma': [0.5, 1.0, 1.5], 'epsilon': [0.0005, 0.001, 0.0015]}
    lj_valid_param_dicts = _make_valid_param_dicts(lj_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.LJ, dict(zip(combos, lj_valid_param_dicts)), {}))

    gauss_arg_dict = {'epsilon': [0.025, 0.05, 0.075], 'sigma': [0.5, 1.0, 1.5]}
    gauss_valid_param_dicts = _make_valid_param_dicts(gauss_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Gauss, dict(zip(combos, gauss_valid_param_dicts)),
                   {}))

    yukawa_arg_dict = {
        'epsilon': [0.00025, 0.0005, 0.00075],
        'kappa': [0.5, 1.0, 1.5]
    }
    yukawa_valid_param_dicts = _make_valid_param_dicts(yukawa_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Yukawa, dict(zip(combos, yukawa_valid_param_dicts)),
                   {}))

    ewald_arg_dict = {"alpha": [0.025, 0.05, 0.075], "kappa": [0.5, 1.0, 1.5]}
    ewald_valid_param_dicts = _make_valid_param_dicts(ewald_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Ewald, dict(zip(combos, ewald_valid_param_dicts)),
                   {}))

    morse_arg_dict = {
        "D0": [0.025, 0.05, 0.075],
        "alpha": [0.5, 1.0, 1.5],
        "r0": [0, 0.05, 0.1]
    }
    morse_valid_param_dicts = _make_valid_param_dicts(morse_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Morse, dict(zip(combos, morse_valid_param_dicts)),
                   {}))

    dpd_conservative_arg_dict = {"A": [0.025, 0.05, 0.075]}
    dpd_conservative_valid_param_dicts = _make_valid_param_dicts(
        dpd_conservative_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.DPDConservative,
                   dict(zip(combos, dpd_conservative_valid_param_dicts)), {}))

    force_shifted_LJ_arg_dict = {
        'sigma': [0.5, 1.0, 1.5],
        'epsilon': [0.0005, 0.001, 0.0015]
    }
    force_shifted_LJ_valid_param_dicts = _make_valid_param_dicts(
        force_shifted_LJ_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.ForceShiftedLJ,
                   dict(zip(combos, force_shifted_LJ_valid_param_dicts)), {}))

    moliere_arg_dict = {
        'qi': [2.5, 7.5, 15],
        'qj': [2, 6, 12],
        'aF': [.134197, .234463, .319536]
    }
    moliere_valid_param_dicts = _make_valid_param_dicts(moliere_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Moliere, dict(zip(combos,
                                             moliere_valid_param_dicts)), {}))

    zbl_arg_dict = {
        'qi': [2.5, 7.5, 15],
        'qj': [2, 6, 12],
        'aF': [.133669, .243535, .341914]
    }
    zbl_valid_param_dicts = _make_valid_param_dicts(zbl_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.ZBL, dict(zip(combos, zbl_valid_param_dicts)), {}))

    mie_arg_dict = {
        'epsilon': [.05, .025, .010],
        'sigma': [.5, 1, 1.5],
        'n': [12, 14, 16],
        'm': [6, 8, 10]
    }
    mie_valid_param_dicts = _make_valid_param_dicts(mie_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Mie, dict(zip(combos, mie_valid_param_dicts)), {}))

    reactfield_arg_dict = {
        'epsilon': [.05, .025, .010],
        'eps_rf': [.5, 1, 1.5],
        'use_charge': [False, True, False]
    }
    reactfield_valid_param_dicts = _make_valid_param_dicts(reactfield_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.ReactionField,
                   dict(zip(combos, reactfield_valid_param_dicts)), {}))

    buckingham_arg_dict = {
        'A': [.05, .025, .010],
        'rho': [.5, 1, 1.5],
        'C': [.05, .025, .01]
    }
    buckingham_valid_param_dicts = _make_valid_param_dicts(buckingham_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Buckingham,
                   dict(zip(combos, buckingham_valid_param_dicts)), {}))

    lj1208_arg_dict = {
        'sigma': [0.5, 1.0, 1.5],
        'epsilon': [0.0005, 0.001, 0.0015]
    }
    lj1208_valid_param_dicts = _make_valid_param_dicts(lj1208_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.LJ1208, dict(zip(combos, lj1208_valid_param_dicts)),
                   {}))

    fourier_arg_dict = {
        'a': [[0.5, 1.0, 1.5], [.05, .1, .15], [.005, .01, .015]],
        'b': [[0.25, 0.034, 0.76], [0.36, 0.12, 0.65], [0.78, 0.04, 0.98]]
    }
    fourier_valid_param_dicts = _make_valid_param_dicts(fourier_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.Fourier, dict(zip(combos,
                                             fourier_valid_param_dicts)), {}))

    slj_arg_dict = {
        'sigma': [0.5, 1.0, 1.5],
        'epsilon': [0.0005, 0.001, 0.0015]
    }
    slj_valid_param_dicts = _make_valid_param_dicts(slj_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.SLJ, dict(zip(combos, slj_valid_param_dicts)), {}))

    dpd_arg_dict = {'A': [0.5, 1.0, 1.5], 'gamma': [0.0005, 0.001, 0.0015]}
    dpd_valid_param_dicts = _make_valid_param_dicts(dpd_arg_dict)
    valid_params_list.append(
        paramtuple(md.pair.DPD, dict(zip(combos, dpd_valid_param_dicts)),
                   {"kT": 2}))

    dpdlj_arg_dict = {
        'sigma': [0.5, 1.0, 1.5],
        'epsilon': [0.0005, 0.001, 0.0015],
        'gamma': [0.034, 33.2, 1.2]
    }
    dpdlj_valid_param_dicts = _make_valid_param_dicts(dpdlj_arg_dict)

    valid_params_list.append(
        paramtuple(md.pair.DPDLJ, dict(zip(combos, dpdlj_valid_param_dicts)),
                   {"kT": 1}))

    dlvo_arg_dict = {
        'kappa': [1.0, 2.0, 5.0],
        'Z': [0.1, 0.5, 2.0],
        'A': [0.1, 0.5, 2.0]
    }
    dlvo_valid_param_dicts = _make_valid_param_dicts(dlvo_arg_dict)

    valid_params_list.append(
        paramtuple(md.pair.DLVO, dict(zip(combos, dlvo_valid_param_dicts)), {}))

    tersoff_arg_dict = {
        'cutoff_thickness': [0.1, 0.5, 1.0],
        'magnitudes': [(0.02, 0.01), (0.0, 0.005), (0.002, 0.003)],
        'exp_factors': [(0.1, 0.1), (0.05, 0.05), (-0.02, 0.04)],
        'lambda3': [0.0, 0.5, 0.3],
        'dimer_r': [1.0, 2.0, 1.2],
        'n': [0.3, 0.5, 0.7],
        'gamma': [0.1, 0.5, 0.4],
        'c': [0.1, 0.5, 2.0],
        'd': [0.1, 0.5, 2.0],
        'm': [0.1, 0.5, 2.0],
        'alpha': [0.1, 0.5, 2.0],
    }
    tersoff_valid_param_dicts = _make_valid_param_dicts(tersoff_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.many_body.Tersoff,
                   dict(zip(combos, tersoff_valid_param_dicts)), {}))

    square_density_arg_dict = {'A': [1.0, 2.0, 5.0], 'B': [0.1, 0.5, 2.0]}
    square_density_valid_param_dicts = _make_valid_param_dicts(
        square_density_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.many_body.SquareDensity,
                   dict(zip(combos, square_density_valid_param_dicts)), {}))

    revcross_arg_dict = {
        'sigma': [1.0, 2.0, 5.0],
        'n': [0.1, 0.5, 2.0],
        'epsilon': [0.1, 0.5, 2.0],
        'lambda3': [0.1, 0.5, 2.0],
    }
    revcross_valid_param_dicts = _make_valid_param_dicts(revcross_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.many_body.RevCross,
                   dict(zip(combos, revcross_valid_param_dicts)), {}))

    opp_arg_dict = {
        'C1': [1.0, 2.0, 5.0],
        'C2': [0.1, 0.5, 2.0],
        'eta1': [15.0, 12.0, 8.0],
        'eta2': [3.0, 2.0, 1.5],
        'k': [1.0, 2.0, 3.0],
        'phi': [0.0, 0.5, np.pi / 2]
    }
    opp_valid_param_dicts = _make_valid_param_dicts(opp_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.pair.OPP, dict(zip(combos, opp_valid_param_dicts)),
                   {}))

    expanded_mie_arg_dict = {
        'epsilon': [.05, .025, .010],
        'sigma': [.5, 1, 1.5],
        'n': [12, 14, 16],
        'm': [6, 8, 10],
        'delta': [.1, .2, .3]
    }
    expanded_mie_valid_param_dicts = _make_valid_param_dicts(
        expanded_mie_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.pair.ExpandedMie,
                   dict(zip(combos, expanded_mie_valid_param_dicts)), {}))

    twf_arg_dict = {
        'sigma': [0.1, 0.2, 0.5],
        'epsilon': [0.1, 0.5, 2.0],
        'alpha': [15.0, 12.0, 8.0]
    }
    twf_valid_param_dicts = _make_valid_param_dicts(twf_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.pair.TWF, dict(zip(combos, twf_valid_param_dicts)),
                   {}))

    cossq_arg_dict = {
        'sigma': [0.5, 1.0, 1.5],
        'epsilon': [0.1, 0.5, 2.0],
        'wca': [True, False, True]
    }
    cossq_valid_param_dicts = _make_valid_param_dicts(cossq_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.pair.CosineSquared,
                    dict(zip(combos, cossq_valid_param_dicts)), {}))

    rs = [
        np.arange(0, 2.6, 0.1),
        np.linspace(0.5, 2.5, 25),
        np.arange(0.8, 2.6, 0.1)
    ]
    Vs = [r[::-1] * 5 for r in rs]
    Fs = [-1 * np.diff(V) / np.diff(r) for V, r in zip(Vs, rs)]
    table_arg_dict = {
        'V': [V[:-1] for V in Vs],
        'F': Fs,
        'r_min': [r[0] for r in rs]
    }
    table_valid_param_dicts = _make_valid_param_dicts(table_arg_dict)
    valid_params_list.append(
        paramtuple(hoomd.md.pair.Table,
                   dict(zip(combos, table_valid_param_dicts)), {}))
    return valid_params_list


@pytest.fixture(scope="function",
                params=_valid_params(),
                ids=(lambda x: x[0].__name__))
def valid_params(request):
    return deepcopy(request.param)


def test_valid_params(valid_params):
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=md.nlist.Cell(),
                                      default_r_cut=2.5)
    for pair in valid_params.pair_potential_params:
        pot.params[pair] = valid_params.pair_potential_params[pair]
    assert _equivalent_data_structures(valid_params.pair_potential_params,
                                       pot.params.to_dict())


def _update_snap(pair_potential, snap):
    if (any(name in str(pair_potential) for name in ['Ewald'])
            and snap.communicator.rank == 0):
        snap.particles.charge[:] = 1.
    if 'SLJ' in str(pair_potential) and snap.communicator.rank == 0:
        snap.particles.diameter[:] = 2
    if 'DLVO' in str(pair_potential) and snap.communicator.rank == 0:
        snap.particles.diameter[0] = 0.2
        snap.particles.diameter[1] = 0.5


def _skip_if_triplet_gpu_mpi(sim, pair_potential):
    """Determines if the simulation is able to run this pair potential."""
    if (isinstance(sim.device, hoomd.device.GPU)
            and sim.device.communicator.num_ranks > 1
            and issubclass(pair_potential, hoomd.md.many_body.Triplet)):
        pytest.skip("Cannot run triplet potentials with GPU+MPI enabled")


def test_attached_params(simulation_factory, lattice_snapshot_factory,
                         valid_params):
    pair_potential, pair_potential_dict, extra_args = valid_params
    pair_keys = valid_params.pair_potential_params.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=md.nlist.Cell(),
                                      default_r_cut=2.5)
    pot.params = valid_params.pair_potential_params

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=10,
                                    a=1.5,
                                    r=0.01)

    _update_snap(valid_params.pair_potential, snap)
    if snap.communicator.rank == 0:
        snap.particles.typeid[:] = np.random.randint(0,
                                                     len(snap.particles.types),
                                                     snap.particles.N)
    sim = simulation_factory(snap)
    _skip_if_triplet_gpu_mpi(sim, valid_params.pair_potential)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(pot)
    sim.run(1)
    assert _equivalent_data_structures(valid_params.pair_potential_params,
                                       pot.params.to_dict())


def test_run(simulation_factory, lattice_snapshot_factory, valid_params):
    pair_keys = valid_params.pair_potential_params.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=md.nlist.Cell(),
                                      default_r_cut=2.5)
    pot.params = valid_params.pair_potential_params

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=7,
                                    a=1.7,
                                    r=0.01)
    if snap.communicator.rank == 0:
        snap.particles.typeid[:] = np.random.randint(0,
                                                     len(snap.particles.types),
                                                     snap.particles.N)
    sim = simulation_factory(snap)
    _skip_if_triplet_gpu_mpi(sim, valid_params.pair_potential)

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(pot)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    sim.operations._schedule()
    old_snap = sim.state.snapshot
    sim.run(2)
    new_snap = sim.state.snapshot
    if new_snap.communicator.rank == 0:
        assert not np.allclose(new_snap.particles.position,
                               old_snap.particles.position)


def test_energy_shifting(simulation_factory, two_particle_snapshot_factory):
    # A subtle bug existed where we used "shifted" instead of "shift" in Python
    # and in C++ we used else if clauses with no error raised if the set Python
    # mode fell through. This means the actual shift mode was not set.
    pytest.skip("Test is broken.")

    def S_r(r, r_cut, r_on):  # noqa: N802 - allow uppercase function name
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

    lj = md.pair.LJ(nlist=md.nlist.Cell(), default_r_cut=r_cut)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=r))

    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    sim.run(0)

    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        E_r = sum(energies)

    snap = sim.state.snapshot
    if snap.communicator.rank == 0:
        snap.particles.position[0] = [0, 0, .1]
        snap.particles.position[1] = [0, 0, r_cut + .1]
    sim.state.snapshot = snap
    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        E_rcut = sum(energies)

    lj_shift = md.pair.LJ(nlist=md.nlist.Cell(),
                          mode='shift',
                          default_r_cut=r_cut)
    lj_shift.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(lj_shift)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    sim.run(0)

    snap = sim.state.snapshot
    if snap.communicator.rank == 0:
        snap.particles.position[0] = [0, 0, .1]
        snap.particles.position[1] = [0, 0, r + .1]
    sim.state.snapshot = snap

    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        assert sum(energies) == E_r - E_rcut

    lj_xplor = md.pair.LJ(nlist=md.nlist.Cell(),
                          mode='xplor',
                          default_r_cut=r_cut)
    lj_xplor.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    lj_xplor.r_on[('A', 'A')] = 0.5
    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(lj_xplor)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))

    sim.operations.integrator = integrator
    sim.run(0)

    energies = sim.operations.integrator.forces[0].energies
    if energies is not None:
        xplor_E = sum(energies)
        assert xplor_E == E_r * S_r(r, r_cut, r_on)

        lj_xplor.r_on[('A', 'A')] = 3.0
        assert sum(energies) == E_r - E_rcut


def _calculate_force(sim):
    """Calculate the forces in a two particle simulation frame.

    Finds the negative derivative of energy divided by inter-particle distance
    """
    dr = 1e-6

    snap = sim.state.snapshot
    if snap.communicator.rank == 0:
        initial_pos = np.array(snap.particles.position)
        snap.particles.position[1, 0] = initial_pos[1, 0] - dr

    sim.state.snapshot = snap
    E0 = sim.operations.integrator.forces[0].energies
    snap = sim.state.snapshot
    if snap.communicator.rank == 0:
        pos = snap.particles.position
        r0 = pos[0] - pos[1]
        mag_r0 = np.linalg.norm(r0)
        direction = r0 / mag_r0

        snap.particles.position[1, 0] = initial_pos[1, 0] + dr

    sim.state.snapshot = snap
    E1 = sim.operations.integrator.forces[0].energies

    snap = sim.state.snapshot
    if snap.communicator.rank == 0:
        pos = snap.particles.position
        mag_r1 = np.linalg.norm(pos[0] - pos[1])
        Fa = -1 * ((sum(E1) - sum(E0)) / (mag_r1 - mag_r0)) * direction
        Fb = -Fa

    snap = sim.state.snapshot
    if snap.communicator.rank == 0:
        snap.particles.position[1] = initial_pos[1]
    sim.state.snapshot = snap
    if sim.state.snapshot.communicator.rank == 0:
        return Fa, Fb
    else:
        return 0, 0  # return dummy values if not on rank 1


def test_force_energy_relationship(simulation_factory,
                                   two_particle_snapshot_factory, valid_params):
    # don't really test DPD and DPDLJ for this test
    pot_name = valid_params.pair_potential.__name__
    if any(pot_name == name for name in ["DPD", "DPDLJ"]):
        pytest.skip("Cannot test force energy relationship for " + pot_name
                    + " pair force")

    pair_keys = valid_params.pair_potential_params.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=md.nlist.Cell(),
                                      default_r_cut=2.5)
    for pair in valid_params.pair_potential_params:
        pot.params[pair] = valid_params.pair_potential_params[pair]

    snap = two_particle_snapshot_factory(particle_types=particle_types, d=1.5)
    _update_snap(valid_params.pair_potential, snap)
    sim = simulation_factory(snap)
    _skip_if_triplet_gpu_mpi(sim, valid_params.pair_potential)
    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(pot)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    sim.run(0)
    for pair in valid_params.pair_potential_params:
        snap = sim.state.snapshot
        if snap.communicator.rank == 0:
            snap.particles.typeid[0] = particle_types.index(pair[0])
            snap.particles.typeid[1] = particle_types.index(pair[1])
        sim.state.snapshot = snap

        calculated_forces = _calculate_force(sim)
        sim_forces = sim.operations.integrator.forces[0].forces
        if sim_forces is not None:
            np.testing.assert_allclose(calculated_forces[0],
                                       sim_forces[0],
                                       rtol=1e-05)
            np.testing.assert_allclose(calculated_forces[1],
                                       sim_forces[1],
                                       rtol=1e-05)


def _forces_and_energies():
    """Return reference force and energy values.

    Reference force and energy values were calculated using Mathematica 12.1.1
    and then stored in the json file below. Values were calculated at
    distances of 0.75 and 1.5 for each argument dictionary
    """
    FEtuple = namedtuple('FEtuple', [
        'pair_potential', 'pair_potential_params', 'extra_args', 'forces',
        'energies'
    ])

    path = Path(__file__).parent / "forces_and_energies.json"

    def json_with_inf(val):
        if isinstance(val, str):
            if val.lower() == "infinity":
                return np.inf
            elif val.lower() == "neg_infinity":
                return -np.inf
        else:
            return val

    with path.open() as f:
        F_and_E = json.load(f)
        param_list = []
        for pot in F_and_E.keys():
            if pot[0].isalpha():
                kT_dict = {'DPD': {'kT': 2}, 'DPDLJ': {'kT': 1}}.get(pot, {})
                for i in range(3):
                    param_list.append(
                        FEtuple(getattr(md.pair, pot),
                                F_and_E[pot]["params"][i], kT_dict, [
                                    json_with_inf(v)
                                    for v in F_and_E[pot]["forces"][i]
                                ], [
                                    json_with_inf(v)
                                    for v in F_and_E[pot]["energies"][i]
                                ]))
    return param_list


def isclose(value, reference, rtol=5e-6):
    """Return if two values are close while automatically managing atol."""
    if isinstance(reference, (Sequence, np.ndarray)):
        ref = np.asarray(reference, np.float64)
        val = np.asarray(reference, np.float64)
        min_value = np.min(np.abs(reference))
        atol = 1e-6 if min_value == 0 else min_value / 1e4
        return np.allclose(val, ref, rtol=rtol, atol=atol, equal_nan=True)
    else:
        atol = 1e-6 if reference == 0 else 0
        return math.isclose(value, reference, rel_tol=rtol, abs_tol=atol)


# We ignore this warning raise by NumPy so we can test the use of infinity in
# some pair potentials currently TWF. This is used when the force from the JSON
# file needs to be multipled by r to compare with the computed force of the
# simulation.
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply")
@pytest.mark.parametrize("forces_and_energies",
                         _forces_and_energies(),
                         ids=lambda x: x.pair_potential.__name__)
def test_force_energy_accuracy(simulation_factory,
                               two_particle_snapshot_factory,
                               forces_and_energies):
    pot_name = forces_and_energies.pair_potential.__name__
    if pot_name == "DPD" or pot_name == "DPDLJ":
        pytest.skip("Cannot test force energy accuracy for " + pot_name
                    + " pair force")

    pot = forces_and_energies.pair_potential(**forces_and_energies.extra_args,
                                             nlist=md.nlist.Cell(),
                                             default_r_cut=2.5)
    pot.params[('A', 'A')] = forces_and_energies.pair_potential_params
    snap = two_particle_snapshot_factory(particle_types=['A'], d=0.75)
    _update_snap(forces_and_energies.pair_potential, snap)
    sim = simulation_factory(snap)
    integrator = md.Integrator(dt=0.005)
    integrator.forces.append(pot)
    integrator.methods.append(
        hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    sim.run(0)
    particle_distances = [0.75, 1.5]
    for i in range(len(particle_distances)):
        d = particle_distances[i]
        r = np.array([0, 0, d]) / d
        snap = sim.state.snapshot
        if snap.communicator.rank == 0:
            snap.particles.position[0] = [0, 0, .1]
            snap.particles.position[1] = [0, 0, d + .1]
        sim.state.snapshot = snap
        sim_energies = sim.operations.integrator.forces[0].energies
        sim_forces = sim.operations.integrator.forces[0].forces
        if sim_energies is not None:
            assert isclose(sum(sim_energies), forces_and_energies.energies[i])
            assert isclose(sim_forces[0], forces_and_energies.forces[i] * r)
            assert isclose(sim_forces[0], -forces_and_energies.forces[i] * r)


# Test logging
@pytest.mark.parametrize(
    'cls, expected_namespace, expected_loggables',
    zip((md.pair.Pair, md.pair.aniso.AnisotropicPair, md.many_body.Triplet),
        (('md', 'pair'), ('md', 'pair', 'aniso'), ('md', 'many_body')),
        itertools.repeat({
            'energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'energies': {
                'category': LoggerCategories.particle,
                'default': True
            },
            'forces': {
                'category': LoggerCategories.particle,
                'default': True
            },
            'torques': {
                'category': LoggerCategories.particle,
                'default': True
            },
            'virials': {
                'category': LoggerCategories.particle,
                'default': True
            },
        })))
def test_logging(cls, expected_namespace, expected_loggables):
    logging_check(cls, expected_namespace, expected_loggables)


def test_pickling(simulation_factory, two_particle_snapshot_factory,
                  valid_params):
    sim = simulation_factory(two_particle_snapshot_factory())
    _skip_if_triplet_gpu_mpi(sim, valid_params.pair_potential)
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=md.nlist.Cell(),
                                      default_r_cut=2.5)
    for pair in valid_params.pair_potential_params:
        pot.params[pair] = valid_params.pair_potential_params[pair]
    pickling_check(pot)
    integrator = hoomd.md.Integrator(0.05, forces=[pot])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(pot)
