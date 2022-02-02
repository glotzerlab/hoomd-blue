# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy

R = 0.9

# test parameters include the class, special pair params, r_cut, force, and energy
special_pair_test_parameters = [(
    hoomd.md.special_pair.LJ,
    dict(epsilon=1.5, sigma=0.5),
    2.5,
    0.0,
    4 * 1.5 * ((0.5 / R)**12 - (0.5 - R)**6),
)]


@pytest.mark.parametrize("special_pair_cls, params, r_cut, force, energy", special_pair_test_parameters)
def test_before_attaching(special_pair_cls, params, r_cut, force, energy):
    potential = special_pair_cls()
    potential.params['A-A'] = params
    potential.r_cut['A-A'] = r_cut
    for key in params:
        assert potential.params['A-A'][key] == pytest.approx(params[key])


@pytest.fixture(scope='session')
def snapshot_factory(two_particle_snapshot_factory):

    def make_snapshot():
        snapshot = two_particle_snapshot_factory(d=R, L=R * 10)
        if snapshot.communicator.rank == 0:
            snapshot.pairs.N = 1
            snapshot.pairs.types = ['A-A']
            snapshot.pairs.typeid[0] = 0
            snapshot.pairs.group[0] = (0, 1)

        return snapshot

    return make_snapshot


@pytest.mark.parametrize("special_pair_cls, params, r_cut, force, energy", special_pair_test_parameters)
def test_after_attaching(snapshot_factory, simulation_factory, special_pair_cls,
                         params, r_cut, force, energy):
    snapshot = snapshot_factory()
    sim = simulation_factory(snapshot)

    potential = special_pair_cls()
    potential.params['A-A'] = params
    potential.r_cut['A-A'] = r_cut

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(potential)

    langevin = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All())
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in params:
        assert potential.params['A-A'][key] == pytest.approx(params[key])


# @pytest.mark.parametrize("bond_cls, potential_kwargs, force, energy",
#                          get_bond_args_forces_and_energies())
# def test_forces_and_energies(two_particle_snapshot_factory, simulation_factory,
#                              bond_cls, potential_kwargs, force, energy):
#     snap = two_particle_snapshot_factory(d=0.969, L=5)
#     if snap.communicator.rank == 0:
#         snap.bonds.N = 1
#         snap.bonds.types = ['bond']
#         snap.bonds.typeid[0] = 0
#         snap.bonds.group[0] = (0, 1)
#         snap.particles.diameter[0] = 0.5
#         snap.particles.diameter[1] = 0.5
#     sim = simulation_factory(snap)

#     bond_potential = bond_cls()
#     bond_potential.params['bond'] = potential_kwargs

#     integrator = hoomd.md.Integrator(dt=0.005)

#     integrator.forces.append(bond_potential)

#     langevin = hoomd.md.methods.Langevin(kT=1,
#                                          filter=hoomd.filter.All(),
#                                          alpha=0.1)
#     integrator.methods.append(langevin)
#     sim.operations.integrator = integrator

#     sim.run(0)

#     sim_energies = sim.operations.integrator.forces[0].energies
#     sim_forces = sim.operations.integrator.forces[0].forces
#     if sim.device.communicator.rank == 0:
#         np.testing.assert_allclose(sum(sim_energies),
#                                    energy,
#                                    rtol=1e-2,
#                                    atol=1e-5)
#         np.testing.assert_allclose(sim_forces[0], [force, 0.0, 0.0],
#                                    rtol=1e-2,
#                                    atol=1e-5)
#         np.testing.assert_allclose(sim_forces[1], [-1 * force, 0.0, 0.0],
#                                    rtol=1e-2,
#                                    atol=1e-5)
