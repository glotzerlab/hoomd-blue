import hoomd
import pytest
import numpy as np
import itertools
import hoomd.hpmc.pytest.conftest

np.random.seed(0)

# Test r_cut
# Test mode
# Test r_on
# Test valid params
# Test invalid params
# Test run
# Test force energy relationship
# Test force
# Test energy


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
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell())
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
    with pytest.raises(TypeError):
        sim.operations.schedule()  # Before setting r_cut

    lj.r_cut[('A', 'A')] = 2.5
    _assert_equivalent_type_params(lj.r_cut.to_dict(), {('A', 'A'): 2.5})
    sim.operations.schedule()
    sim.run(1)
    _assert_equivalent_type_params(lj.r_cut.to_dict(), {('A', 'A'): 2.5})


@pytest.mark.parametrize("mode", ['none', 'shifted', 'xplor'])
def test_mode(simulation_factory, two_particle_snapshot_factory, mode):
    cell = hoomd.md.nlist.Cell()
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode=1)
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode='str')
    with pytest.raises(hoomd.typeconverter.TypeConversionError):
        lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode=[1, 2, 3])

    lj = hoomd.md.pair.LJ(nlist=cell, r_cut=2.5, mode=mode)
    lj.params[('A', 'A')] = {'sigma': 1, 'epsilon': 0.5}
    sim = simulation_factory(two_particle_snapshot_factory(dimensions=3, d=.5))
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    sim.operations.schedule()
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

    sim.operations.schedule()  # Before setting r_on
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 0.0})

    lj.r_on[('A', 'A')] = 1.5
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 1.5})
    sim.run(1)
    _assert_equivalent_type_params(lj.r_on.to_dict(), {('A', 'A'): 1.5})


def test_valid_params(valid_params):
    pair_potential, pair_potential_dict = valid_params[1:]
    pot = pair_potential(nlist=hoomd.md.nlist.Cell())
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]
    _assert_equivalent_type_params(pot.params.to_dict(), pair_potential_dict)


def test_invalid_params(invalid_params):
    pair_potential, pair_potential_dict = invalid_params[1:]
    pot = pair_potential(nlist=hoomd.md.nlist.Cell())
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
    pair_potential, pair_potential_dict = valid_params[1:]
    pair_keys = pair_potential_dict.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = pair_potential(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=10, a=1.5, r=0.01)
    if snap.exists:
        snap.particles.typeid[:] = np.random.randint(0,
                                                     len(snap.particles.types),
                                                     snap.particles.N)
    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator.forces.append(pot)
    sim.operations.schedule()
    sim.run(10)
    attached_pot = sim.operations.integrator.forces[0]
    _assert_equivalent_type_params(attached_pot.params.to_dict(),
                                   pair_potential_dict)


@pytest.mark.parametrize("nsteps", [3, 5, 10])
def test_run(simulation_factory, lattice_snapshot_factory,
             valid_params, nsteps):
    pair_potential, pair_potential_dict = valid_params[1:]
    pair_keys = pair_potential_dict.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = pair_potential(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]

    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=2, a=5, r=0.01)
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
    initial_pos = sim.state.snapshot.particles.position
    sim.run(nsteps)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(sim.state.snapshot.particles.position,
                                   initial_pos)


# This function calculates the forces in a two particle simulation frame by
# finding the negative derivative of energy over inter particle distance
def _calculate_force(sim):
    snap = sim.state.snapshot
    initial_pos = snap.particles.position
    snap.particles.position[1] = initial_pos[1] * 0.99999999
    sim.state.snapshot = snap
    E0 = sim.operations.integrator.forces[0].energies
    pos = sim.state.snapshot.particles.position
    r0 = pos[0] - pos[1]
    mag_r0 = np.linalg.norm(r0)
    direction = r0 / mag_r0

    snap = sim.state.snapshot
    snap.particles.position[1] = initial_pos[1] * 1.00000001
    sim.state.snapshot = snap
    E1 = sim.operations.integrator.forces[0].energies
    pos = sim.state.snapshot.particles.position
    mag_r1 = np.linalg.norm(pos[0] - pos[1])

    Fa = -1 * ((E1[0] - E0[0]) / (mag_r1 - mag_r0)) * 2 * direction
    Fb = -1 * ((E1[1] - E0[1]) / (mag_r1 - mag_r0)) * 2 * direction * -1
    snap = sim.state.snapshot
    snap.particles.position[1] = initial_pos[1]
    sim.state.snapshot = snap
    return Fa, Fb


@pytest.mark.parametrize("nsteps", [1, 5, 10])
def test_force_energy_relationship(simulation_factory,
                                   two_particle_snapshot_factory,
                                   valid_params, nsteps):
    pair_potential, pair_potential_dict = valid_params[1:]
    pair_keys = pair_potential_dict.keys()
    particle_types = list(set(itertools.chain.from_iterable(pair_keys)))
    pot = pair_potential(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    for pair in pair_potential_dict:
        pot.params[pair] = pair_potential_dict[pair]

    snap = two_particle_snapshot_factory(particle_types=particle_types, d=1.5)
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
        sim_forces = sim.operations.integrator.forces[0].forces
        np.testing.assert_allclose(calculated_forces[0], sim_forces[0])
        np.testing.assert_allclose(calculated_forces[1], sim_forces[1])
