import hoomd
import pytest
import numpy as np
import itertools
from copy import deepcopy
import json
from pathlib import Path
from collections import namedtuple


def _assert_equivalent_type_params(type_param1, type_param2):
    """
    Compare entries in type_param1 and type_param2.

    type_param1 is the dictionary used to set the potential
    arguments, whereas type_param2 is the dictionary returned
    from the potential's to_dict method. This means type_param2
    includes default arguments in addition to all keys in type_param1
    """
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
    for invalid_mode in [1, 'str', [1, 2, 3]]:
        with pytest.raises(hoomd.typeconverter.TypeConversionError):
            lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode=invalid_mode)


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
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 1.5})
    sim.operations.schedule()
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 1.5})

    lj.r_on[('A', 'A')] = 1.0
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 1.0})


def _make_invalid_param_dict(valid_dict):
    invalid_dicts = [valid_dict] * len(valid_dict.keys()) * 2
    count = 0
    for key in valid_dict.keys():
        # Set one invalid argument per dictionary
        # Set two invalid arguments per key
        if not isinstance(invalid_dicts[count][key], list):
            invalid_dicts[count][key] = [1, 2]
            invalid_dicts[count + 1][key] = 'str'
        else:
            invalid_dicts[count][key] = 1
            invalid_dicts[count + 1][key] = False
        count += 2
    return invalid_dicts

paramtuple = namedtuple('paramtuple',
                        ['pair_potential',
                         'pair_potential_params',
                         'extra_args'])


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
    invalid_params_list.extend(_make_invalid_params(lj_invalid_dicts,
                                                    hoomd.md.pair.LJ,
                                                    {}))

    gauss_valid_dict = {'sigma': 0.05, 'epsilon': 0.05}
    gauss_invalid_dicts = _make_invalid_param_dict(gauss_valid_dict)
    invalid_params_list.extend(_make_invalid_params(gauss_invalid_dicts,
                                                    hoomd.md.pair.Gauss,
                                                    {}))

    yukawa_valid_dict = {"epsilon": 0.0005, "kappa": 1}
    yukawa_invalid_dicts = _make_invalid_param_dict(yukawa_valid_dict)
    invalid_params_list.extend(_make_invalid_params(yukawa_invalid_dicts,
                                                    hoomd.md.pair.Yukawa,
                                                    {}))

    ewald_valid_dict = {"alpha": 0.05, "kappa": 1}
    ewald_invalid_dicts = _make_invalid_param_dict(ewald_valid_dict)
    invalid_params_list.extend(_make_invalid_params(ewald_invalid_dicts,
                                                    hoomd.md.pair.Ewald,
                                                    {}))

    morse_valid_dict = {"D0": 0.05, "alpha": 1, "r0": 0}
    morse_invalid_dicts = _make_invalid_param_dict(morse_valid_dict)
    invalid_params_list.extend(_make_invalid_params(morse_invalid_dicts,
                                                    hoomd.md.pair.Morse,
                                                    {}))

    dpd_conservative_valid_dict = {"A": 0.05}
    dpd_conservative_invalid_dicts = _make_invalid_param_dict(dpd_conservative_valid_dict)
    invalid_params_list.extend(_make_invalid_params(dpd_conservative_invalid_dicts,
                                                    hoomd.md.pair.DPDConservative,
                                                    {}))

    force_shifted_LJ_valid_dict = {"epsilon": 0.0005, "sigma": 1}
    force_shifted_LJ_invalid_dicts = _make_invalid_param_dict(force_shifted_LJ_valid_dict)
    invalid_params_list.extend(_make_invalid_params(force_shifted_LJ_invalid_dicts,
                                                    hoomd.md.pair.ForceShiftedLJ,
                                                    {}))

    moliere_valid_dict = {"qi": 15, "qj": 12, "aF": 1}
    moliere_invalid_dicts = _make_invalid_param_dict(moliere_valid_dict)
    invalid_params_list.extend(_make_invalid_params(moliere_invalid_dicts,
                                                    hoomd.md.pair.Moliere,
                                                    {}))
    zbl_valid_dict = {"qi": 10, "qj": 8, "aF": 0.5}
    zbl_invalid_dicts = _make_invalid_param_dict(zbl_valid_dict)
    invalid_params_list.extend(_make_invalid_params(zbl_invalid_dicts,
                                                    hoomd.md.pair.ZBL,
                                                    {}))

    mie_valid_dict = {"epsilon": 0.05, "sigma": 0.5, "n": 12, "m": 6}
    mie_invalid_dicts = _make_invalid_param_dict(mie_valid_dict)
    invalid_params_list.extend(_make_invalid_params(mie_invalid_dicts,
                                                    hoomd.md.pair.Mie,
                                                    {}))

    rf_valid_dict = {"epsilon": 0.05, "eps_rf": 0.5, "use_charge": False}
    rf_invalid_dicts = _make_invalid_param_dict(rf_valid_dict)
    invalid_params_list.extend(_make_invalid_params(rf_invalid_dicts,
                                                    hoomd.md.pair.ReactionField,
                                                    {}))

    buckingham_valid_dict = {"A": 0.05, "rho": 0.5, "C": 0.05}
    buckingham_invalid_dicts = _make_invalid_param_dict(buckingham_valid_dict)
    invalid_params_list.extend(_make_invalid_params(buckingham_invalid_dicts,
                                                    hoomd.md.pair.Buckingham,
                                                    {}))

    lj1208_valid_dict = {"sigma": 0.5, "epsilon": 0.0005}
    lj1208_invalid_dicts = _make_invalid_param_dict(lj1208_valid_dict)
    invalid_params_list.extend(_make_invalid_params(lj1208_invalid_dicts,
                                                    hoomd.md.pair.LJ1208,
                                                    {}))

    fourier_valid_dict = {"a": [0.5, 1.0, 1.5], "b": [0.25, 0.034, 0.76]}
    fourier_invalid_dicts = _make_invalid_param_dict(fourier_valid_dict)
    invalid_params_list.extend(_make_invalid_params(fourier_invalid_dicts,
                                                    hoomd.md.pair.Fourier,
                                                    {}))

    slj_valid_dict = {"sigma": 0.5, "epsilon": 0.0005}
    slj_invalid_dicts = _make_invalid_param_dict(slj_valid_dict)
    invalid_params_list.extend(_make_invalid_params(slj_invalid_dicts,
                                                    hoomd.md.pair.SLJ,
                                                    {}))

    dpd_valid_dict = {"A": 0.5, "gamma": 0.0005}
    dpd_invalid_dicts = _make_invalid_param_dict(dpd_valid_dict)
    invalid_params_list.extend(_make_invalid_params(dpd_invalid_dicts,
                                                    hoomd.md.pair.DPD,
                                                    {'kT': 2}))

    dpdlj_valid_dict = {'sigma': 0.5, 'epsilon': 0.0005, 'gamma': 0.034}
    dpdlj_invalid_dicts = _make_invalid_param_dict(dpdlj_valid_dict)
    invalid_params_list.extend(_make_invalid_params(dpdlj_invalid_dicts,
                                                    hoomd.md.pair.DPDLJ,
                                                    {'kT': 1}))

    dlvo_valid_dict = {'kappa': 1.0, 'Z': 0.1, 'A': 0.1}
    dlvo_invalid_dicts = _make_invalid_param_dict(dlvo_valid_dict)
    invalid_params_list.extend(_make_invalid_params(dlvo_invalid_dicts,
                                                    hoomd.md.pair.DLVO,
                                                    {}))
    return invalid_params_list


@pytest.fixture(scope="function", params=_invalid_params(), ids=(lambda x: x[0].__name__))
def invalid_params(request):
    return deepcopy(request.param)


def test_invalid_params(invalid_params):
    pot = invalid_params.pair_potential(**invalid_params.extra_args,
                                        nlist=hoomd.md.nlist.Cell(),
                                        mode='none')
    for pair in invalid_params.pair_potential_params:
        if isinstance(pair, tuple):
            with pytest.raises(hoomd.typeconverter.TypeConversionError):
                pot.params[pair] = invalid_params.pair_potential_params[pair]


def test_invalid_pair_key():
    pot = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell())
    for invalid_key in [3, [1, 2], 'str']:
        with pytest.raises(KeyError):
            pot.r_cut[invalid_key] = 2.5


def _make_valid_param_dicts(arg_dict):
    """
    Unpack dictionary of lists of numbers into dictionary of numbers.

    Ex: turn {'a': [0, 1], 'b':[2, 3]} into [{'a': 0, 'b': 2}, {'a': 1, 'b': 3}]
    """
    return [dict(zip(arg_dict, val)) for val in zip(*arg_dict.values())]


def _valid_params(particle_types=['A', 'B']):
    valid_params_list = []
    combos = list(itertools.combinations_with_replacement(particle_types, 2))
    lj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                   'epsilon': [0.0005, 0.001, 0.0015]}
    lj_valid_param_dicts = _make_valid_param_dicts(lj_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.LJ,
                                        dict(zip(combos,
                                                 lj_valid_param_dicts)),
                                        {}))

    gauss_arg_dict = {'epsilon': [0.025, 0.05, 0.075],
                      'sigma': [0.5, 1.0, 1.5]}
    gauss_valid_param_dicts = _make_valid_param_dicts(gauss_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Gauss,
                                        dict(zip(combos,
                                                 gauss_valid_param_dicts)),
                                        {}))

    yukawa_arg_dict = {'epsilon': [0.00025, 0.0005, 0.00075],
                       'kappa': [0.5, 1.0, 1.5]}
    yukawa_valid_param_dicts = _make_valid_param_dicts(yukawa_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Yukawa,
                                        dict(zip(combos,
                                                 yukawa_valid_param_dicts)),
                                        {}))

    ewald_arg_dict = {"alpha": [0.025, 0.05, 0.075],
                      "kappa": [0.5, 1.0, 1.5]}
    ewald_valid_param_dicts = _make_valid_param_dicts(ewald_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Ewald,
                                        dict(zip(combos,
                                                 ewald_valid_param_dicts)),
                                        {}))

    morse_arg_dict = {"D0": [0.025, 0.05, 0.075],
                      "alpha": [0.5, 1.0, 1.5],
                      "r0": [0, 0.05, 0.1]}
    morse_valid_param_dicts = _make_valid_param_dicts(morse_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Morse,
                                        dict(zip(combos,
                                                 morse_valid_param_dicts)),
                                        {}))

    dpd_conservative_arg_dict = {"A": [0.025, 0.05, 0.075]}
    dpd_conservative_valid_param_dicts = _make_valid_param_dicts(dpd_conservative_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.DPDConservative,
                                        dict(zip(combos,
                                                 dpd_conservative_valid_param_dicts)),
                                        {}))

    force_shifted_LJ_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                                 'epsilon': [0.0005, 0.001, 0.0015]}
    force_shifted_LJ_valid_param_dicts = _make_valid_param_dicts(force_shifted_LJ_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.ForceShiftedLJ,
                                        dict(zip(combos,
                                                 force_shifted_LJ_valid_param_dicts)),
                                        {}))

    moliere_arg_dict = {'qi': [2.5, 7.5, 15], 'qj': [2, 6, 12],
                        'aF': [.134197, .234463, .319536]}
    moliere_valid_param_dicts = _make_valid_param_dicts(moliere_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Moliere,
                                        dict(zip(combos,
                                                 moliere_valid_param_dicts)),
                                        {}))

    zbl_arg_dict = {'qi': [2.5, 7.5, 15], 'qj': [2, 6, 12],
                    'aF': [.133669, .243535, .341914]}
    zbl_valid_param_dicts = _make_valid_param_dicts(zbl_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.ZBL,
                                        dict(zip(combos,
                                                 zbl_valid_param_dicts)),
                                        {}))

    mie_arg_dict = {'epsilon': [.05, .025, .010], 'sigma': [.5, 1, 1.5],
                    'n': [12, 14, 16], 'm': [6, 8, 10]}
    mie_valid_param_dicts = _make_valid_param_dicts(mie_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Mie,
                                        dict(zip(combos,
                                                 mie_valid_param_dicts)),
                                        {}))

    reactfield_arg_dict = {'epsilon': [.05, .025, .010], 'eps_rf': [.5, 1, 1.5],
                           'use_charge': [False, True, False]}
    reactfield_valid_param_dicts = _make_valid_param_dicts(reactfield_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.ReactionField,
                                        dict(zip(combos,
                                                 reactfield_valid_param_dicts)),
                                        {}))

    buckingham_arg_dict = {'A': [.05, .025, .010], 'rho': [.5, 1, 1.5],
                           'C': [.05, .025, .01]}
    buckingham_valid_param_dicts = _make_valid_param_dicts(buckingham_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Buckingham,
                                        dict(zip(combos,
                                                 buckingham_valid_param_dicts)),
                                        {}))

    lj1208_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                       'epsilon': [0.0005, 0.001, 0.0015]}
    lj1208_valid_param_dicts = _make_valid_param_dicts(lj1208_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.LJ1208,
                                        dict(zip(combos,
                                                 lj1208_valid_param_dicts)),
                                        {}))

    fourier_arg_dict = {'a': [[0.5, 1.0, 1.5],
                              [.05, .1, .15],
                              [.005, .01, .015]],
                        'b': [[0.25, 0.034, 0.76],
                              [0.36, 0.12, 0.65],
                              [0.78, 0.04, 0.98]]}
    fourier_valid_param_dicts = _make_valid_param_dicts(fourier_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Fourier,
                                        dict(zip(combos,
                                                 fourier_valid_param_dicts)),
                                        {}))

    slj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                    'epsilon': [0.0005, 0.001, 0.0015]}
    slj_valid_param_dicts = _make_valid_param_dicts(slj_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.SLJ,
                                        dict(zip(combos,
                                                 slj_valid_param_dicts)),
                                        {}))

    dpd_arg_dict = {'A': [0.5, 1.0, 1.5],
                    'gamma': [0.0005, 0.001, 0.0015]}
    dpd_valid_param_dicts = _make_valid_param_dicts(dpd_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.DPD,
                                        dict(zip(combos,
                                                 dpd_valid_param_dicts)),
                                        {"kT": 2}))

    dpdlj_arg_dict = {'sigma': [0.5, 1.0, 1.5],
                      'epsilon': [0.0005, 0.001, 0.0015],
                      'gamma': [0.034, 33.2, 1.2]}
    dpdlj_valid_param_dicts = _make_valid_param_dicts(dpdlj_arg_dict)

    valid_params_list.append(paramtuple(hoomd.md.pair.DPDLJ,
                                        dict(zip(combos,
                                                 dpdlj_valid_param_dicts)),
                                        {"kT": 1}))

    dlvo_arg_dict = {'kappa': [1.0, 2.0, 5.0],
                     'Z': [0.1, 0.5, 2.0],
                     'A': [0.1, 0.5, 2.0]}
    dlvo_valid_param_dicts = _make_valid_param_dicts(dlvo_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.DLVO,
                                        dict(zip(combos,
                                                 dlvo_valid_param_dicts)),
                                        {}))

    tersoff_arg_dict = {
            'cutoff_thickness': [0.1, 0.5, 1.0],
            'C1': [1.0, 2.0, 5.0],
            'C2': [0.1, 0.5, 2.0],
            'lambda1': [0.1, 0.5, 2.0],
            'lambda2': [0.1, 0.5, 2.0],
            'lambda3': [0.0, 0.5, 2.0],
            'dimer_r': [1.0, 2.0, 2.5],
            'n': [0.0, 0.5, 2.0],
            'gamma': [0.1, 0.5, 2.0],
            'c': [0.1, 0.5, 2.0],
            'd': [0.1, 0.5, 2.0],
            'm': [0.1, 0.5, 2.0],
            'alpha': [0.1, 0.5, 2.0],
            }
    tersoff_valid_param_dicts = _make_valid_param_dicts(tersoff_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.Tersoff,
                                        dict(zip(combos,
                                                 tersoff_valid_param_dicts)),
                                        {}))

    square_density_arg_dict = {'A': [1.0, 2.0, 5.0], 'B': [0.1, 0.5, 2.0]}
    square_density_valid_param_dicts = _make_valid_param_dicts(
            square_density_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.SquareDensity,
                                        dict(zip(combos,
                                                 square_density_valid_param_dicts)),
                                        {}))

    revcross_arg_dict = {
            'sigma': [1.0, 2.0, 5.0],
            'n': [0.1, 0.5, 2.0],
            'epsilon': [0.1, 0.5, 2.0],
            'lambda3': [0.1, 0.5, 2.0],
            }
    revcross_valid_param_dicts = _make_valid_param_dicts(revcross_arg_dict)
    valid_params_list.append(paramtuple(hoomd.md.pair.RevCross,
                                        dict(zip(combos,
                                                 revcross_valid_param_dicts)),
                                        {}))
    return valid_params_list


@pytest.fixture(scope="function", params=_valid_params(), ids=(lambda x: x[0].__name__))
def valid_params(request):
    return deepcopy(request.param)

def test_valid_params(valid_params):
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=hoomd.md.nlist.Cell(),
                                      r_cut=2.5,
                                      mode='none')
    for pair in valid_params.pair_potential_params:
        pot.params[pair] = valid_params.pair_potential_params[pair]
    _assert_equivalent_type_params(valid_params.pair_potential_params,
                                   pot.params.to_dict())


def _update_snap(pair_potential, snap):
    if 'Ewald' in str(pair_potential) and snap.exists:
        snap.particles.charge[:] = 1
    elif 'SLJ' in str(pair_potential) and snap.exists:
        snap.particles.diameter[:] = 2
    elif 'DLVO' in str(pair_potential) and snap.exists:
        snap.particles.diameter[0] = 0.2
        snap.particles.diameter[1] = 0.5


def test_attached_params(simulation_factory, lattice_snapshot_factory,
                         valid_params):
    pair_potential, pair_potential_dict, extra_args = valid_params
    pair_keys = valid_params.pair_potential_params.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=hoomd.md.nlist.Cell(),
                                      r_cut=2.5, mode='none')
    pot.params = valid_params.pair_potential_params

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=10, a=1.5, r=0.01)

    _update_snap(valid_params.pair_potential, snap)
    if snap.exists:
        snap.particles.typeid[:] = np.random.randint(0,
                                                     len(snap.particles.types),
                                                     snap.particles.N)
    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(pot)
    sim.run(1)
    _assert_equivalent_type_params(pot.params.to_dict(),
                                   valid_params.pair_potential_params)


def test_run(simulation_factory, lattice_snapshot_factory, valid_params):
    pair_keys = valid_params.pair_potential_params.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=hoomd.md.nlist.Cell(),
                                      r_cut=2.5, mode='none')
    pot.params = valid_params.pair_potential_params

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=7, a=1.7, r=0.01)
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
            assert not np.allclose(new_snap.particles.position,
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

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=r))

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


def _calculate_force(sim):
    """
    Calculate the forces in a two particle simulation frame.

    Finds the negative derivative of energy divided by inter-particle distance
    """
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
    pot_name = valid_params.pair_potential.__name__
    if pot_name == "DPD" or pot_name == "DPDLJ":
        pytest.skip("Cannot test force energy relationship for " +
                    pot_name + " pair force")

    pair_keys = valid_params.pair_potential_params.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = valid_params.pair_potential(**valid_params.extra_args,
                                      nlist=hoomd.md.nlist.Cell(),
                                      r_cut=2.5, mode='none')
    for pair in valid_params.pair_potential_params:
        pot.params[pair] = valid_params.pair_potential_params[pair]

    snap = two_particle_snapshot_factory(particle_types=particle_types, d=1.5)
    _update_snap(valid_params.pair_potential, snap)
    sim = simulation_factory(snap)
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(pot)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.operations.schedule()
    for pair in valid_params.pair_potential_params:
        snap = sim.state.snapshot
        if snap.exists:
            snap.particles.typeid[0] = particle_types.index(pair[0])
            snap.particles.typeid[1] = particle_types.index(pair[1])
        sim.state.snapshot = snap

        calculated_forces = _calculate_force(sim)
        sim_forces = sim.operations.integrator.forces[0].forces
        if sim_forces is not None:
            np.testing.assert_allclose(calculated_forces[0],
                                       sim_forces[0],
                                       rtol=1e-06)
            np.testing.assert_allclose(calculated_forces[1],
                                       sim_forces[1],
                                       rtol=1e-06)

FandEtuple = namedtuple('FandEtuple',
                        ['pair_potential',
                         'pair_potential_params',
                         'extra_args',
                         'forces',
                         'energies'])


def _forces_and_energies():
    """
    Return reference force and energy values.

    Reference force and energy values were calculated using Mathematica 12.1.1
    and then stored in the json file below. Values were calculated at
    distances of 0.75 and 1.5 for each argument dictionary
    """
    path = Path(__file__).parent / "forces_and_energies.json"
    with path.open() as f:
        F_and_E = json.load(f)
        param_list = []
        for pot in F_and_E.keys():
            if pot[0].isalpha():
                kT_dict = {}
                if pot == "DPD":
                    kT_dict = {"kT": 2}
                elif pot == "DPDLJ":
                    kT_dict = {"kT": 1}
                for i in range(3):
                    param_list.append(FandEtuple(getattr(hoomd.md.pair, pot),
                                                 F_and_E[pot]["params"][i],
                                                 kT_dict,
                                                 F_and_E[pot]["forces"][i],
                                                 F_and_E[pot]["energies"][i]))
    return param_list


@pytest.fixture(scope="function",
                params=_forces_and_energies(),
                ids=(lambda x: x[0].__name__))
def forces_and_energies(request):
    return deepcopy(request.param)


def test_force_energy_accuracy(simulation_factory,
                               two_particle_snapshot_factory,
                               forces_and_energies):
    pot = forces_and_energies.pair_potential(**forces_and_energies.extra_args,
                                             nlist=hoomd.md.nlist.Cell(),
                                             r_cut=2.5, mode='none')
    pot.params[('A', 'A')] = forces_and_energies.pair_potential_params
    snap = two_particle_snapshot_factory(particle_types=['A'], d=0.75)
    _update_snap(forces_and_energies.pair_potential, snap)
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
            if forces_and_energies.energies[i] == 0 or sum(sim_energies) == 0:
                atol = 1e-06
            np.testing.assert_allclose(forces_and_energies.energies[i],
                                       sum(sim_energies),
                                       rtol=5e-06,
                                       atol=atol)
        if sim_forces is not None:
            if forces_and_energies.forces[i] == 0 or sum(sim_forces[0]) == 0:
                atol = 1e-06
            np.testing.assert_allclose(forces_and_energies.forces[i] * r,
                                       sim_forces[0],
                                       rtol=5e-06,
                                       atol=atol)
            np.testing.assert_allclose(forces_and_energies.forces[i] * r * -1,
                                       sim_forces[1],
                                       rtol=5e-06,
                                       atol=atol)
