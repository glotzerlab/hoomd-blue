import hoomd
import pytest
import numpy as np
import itertools
from copy import deepcopy

np.random.seed(0)


def assert_equivalent_type_params(type_param1, type_param2):
    for pair in type_param1:
        if isinstance(type_param1[pair], dict):
            for key in type_param1[pair]:
                np.testing.assert_allclose(type_param1[pair][key],
                                           type_param2[pair][key])
        else:
            assert type_param1[pair] == type_param2[pair]


def assert_equivalent_parameter_dicts(param_dict1, param_dict2):
    for key in param_dict1:
        assert param_dict1[key] == param_dict2[key]


def _lj_valid_params():
    particle_types_list = [['A'], ['A', 'B'],
                           ['A', 'B', 'C'],
                           ['A', 'B', 'C', 'D']]
    combos = []
    for particle_types in particle_types_list:
        type_combo = list(itertools.combinations_with_replacement(particle_types,
                                                                  2))
        combos.append(type_combo)
        N = len(type_combo)
    sample_range = np.linspace(0.5, 1.5, 100)
    samples = np.array_split(np.random.choice(sample_range,
                                              size=N * len(combos) * 3,
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


@pytest.fixture(scope="function", params=_lj_valid_params())
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

_lj_invalid_params = [(hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': [1, 2], 'epsilon': 1.0}},
                       {('A', 'A'): 2.5},
                       {('A', 'A'): 2.1},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 1.0, 'epsilon': [1, 2]}},
                       {('A', 'A'): 2.5},
                       {('A', 'A'): 2.1},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 'str', 'epsilon': 1.0}},
                       {('A', 'A'): 2.5},
                       {('A', 'A'): 2.1},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 1.0, 'epsilon': 'str'}},
                       {('A', 'A'): 2.5},
                       {('A', 'A'): 2.1},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 1.0, 'epsilon': 1.0}},
                       {('A', 'A'): [1, 2]},
                       {('A', 'A'): 2.1},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 1.0, 'epsilon': 1.0}},
                       {('A', 'A'): 'str'},
                       {('A', 'A'): 2.1},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 1.0, 'epsilon': 1.0}},
                       {('A', 'A'): 2.5},
                       {('A', 'A'): [1, 2]},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 1.0, 'epsilon': 1.0}},
                       {('A', 'A'): 2.5},
                       {('A', 'A'): 'str'},
                       'none'),
                      (hoomd.md.pair.LJ,
                       {('A', 'A'): {'sigma': 1.0, 'epsilon': 1.0}},
                       {('A', 'A'): 2.5},
                       {('A', 'A'): 2.1},
                       5),
                      (hoomd.md.pair.LJ,
                       {3: {'sigma': 1.0, 'epsilon': 1.0}},
                       {3: 2.5},
                       {3: 2.1},
                       'none')]


@pytest.fixture(scope="function", params=_lj_invalid_params)
def invalid_params(request):
    return deepcopy(request.param)


def test_invalid_params(invalid_params):
    pair_potential, pair_potential_dict, r_cut, r_on, mode = invalid_params
    cell = hoomd.md.nlist.Cell()
    if isinstance(mode, str):
        pot = pair_potential(nlist=cell, mode=mode)
        for pair in pair_potential_dict:
            if isinstance(pair, tuple):
                with pytest.raises(hoomd.typeconverter.TypeConversionError):
                    pot.params[pair] = pair_potential_dict[pair]
                    pot.r_cut[pair] = r_cut[pair]
                    pot.r_on[pair] = r_on[pair]
            else:
                with pytest.raises(KeyError):
                    pot.params[pair] = pair_potential_dict[pair]
                    pot.r_cut[pair] = r_cut[pair]
                    pot.r_on[pair] = r_on[pair]
    else:
        with pytest.raises(hoomd.typeconverter.TypeConversionError):
            pot = pair_potential(nlist=cell, mode=mode)


def test_attached_params(simulation_factory, lattice_snapshot_factory,
                         valid_params):
    pair_potential, pair_potential_dict, r_cut, r_on, mode = valid_params
    particle_types = list(set(itertools.chain.from_iterable(r_cut.keys())))
    cell = hoomd.md.nlist.Cell()
    pot = pair_potential(nlist=cell, r_cut=2.5)

    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]
        pot.r_cut[pair] = r_cut[pair]
        pot.r_on[pair] = r_on[pair]

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=100, a=0.5, r=0.01)
    snap.particles.typeid[:] = np.random.randint(0, len(snap.particles.types),
                                                 snap.particles.N)
    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(pot)
    sim.operations.schedule()
    attached_pot = sim.operations.integrator.forces[0]
    assert_equivalent_type_params(attached_pot.params.to_dict(),
                                  pair_potential_dict)
    assert_equivalent_type_params(attached_pot.r_cut.to_dict(), r_cut)
    assert_equivalent_type_params(attached_pot.r_on.to_dict(), r_on)
    assert_equivalent_parameter_dicts(attached_pot.nlist._param_dict,
                                      cell._param_dict)
