import hoomd
import pytest
import numpy as np
import itertools
import hoomd.hpmc.pytest.conftest

np.random.seed(0)


def test_attach(simulation_factory, two_particle_snapshot_factory, pair_and_params):
    pair = pair_and_params[0]
    params = pair_and_params[1]

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.5))
    integrator = hoomd.md.Integrator(dt=0.5)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1, seed=1))
    pair.params[('A', 'A')] = params
    pair.r_cut[('A', 'A')] = .02
    integrator.forces.append(pair)
    sim.operations.integrator = integrator
    sim.operations.schedule()
    sim.run(10)


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


def test_valid_params(valid_params):
    pair_potential, pair_potential_dict, r_cut, r_on, mode = valid_params
    cell = hoomd.md.nlist.Cell()
    if mode is not None:
        pot = pair_potential(nlist=cell, mode=mode)
    else:
        pot = pair_potential(nlist=cell)
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]
        pot.r_cut[pair] = r_cut[pair]
        if r_on is not None:
            pot.r_on[pair] = r_on[pair]

    assert_equivalent_type_params(pot.params.to_dict(), pair_potential_dict)
    assert_equivalent_type_params(pot.r_cut.to_dict(), r_cut)
    assert_equivalent_type_params(pot.r_on.to_dict(), r_on)
    assert_equivalent_parameter_dicts(pot.nlist._param_dict, cell._param_dict)


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
    if mode is not None:
        pot = pair_potential(nlist=cell, r_cut=2.5, mode=mode)
    else:
        pot = pair_potential(nlist=cell, r_cut=2.5)

    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]
        pot.r_cut[pair] = r_cut[pair]
        if r_on is not None:
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
    if r_on is None:
        assert attached_pot.r_on.to_dict() == {key: 0.0 for key in r_cut.keys()}
    else:
        assert_equivalent_type_params(attached_pot.r_on.to_dict(), r_on)
    assert_equivalent_parameter_dicts(attached_pot.nlist._param_dict,
                                      cell._param_dict)
