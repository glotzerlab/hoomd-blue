# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy as np

_harmonic_args = {'k': [30.0, 25.0, 20.0], 'r0': [1.6, 1.7, 1.8]}
_harmonic_arg_list = [(hoomd.md.bond.Harmonic, dict(zip(_harmonic_args, val)))
                      for val in zip(*_harmonic_args.values())]

_FENE_args = {
    'k': [30.0, 25.0, 20.0],
    'r0': [1.6, 1.7, 1.8],
    'epsilon': [0.9, 1.0, 1.1],
    'sigma': [1.1, 1.0, 0.9]
}
_FENE_arg_list = [(hoomd.md.bond.FENE, dict(zip(_FENE_args, val)))
                  for val in zip(*_FENE_args.values())]

_Tether_args = {
    'k_b': [5.0, 6.0, 7.0],
    'l_min': [0.7, 0.8, 0.9],
    'l_c1': [0.9, 1.05, 1.1],
    'l_c0': [1.1, 1.1, 1.3],
    'l_max': [1.3, 1.3, 1.5]
}
_Tether_arg_list = [(hoomd.md.bond.Tether, dict(zip(_Tether_args, val)))
                    for val in zip(*_Tether_args.values())]


def get_bond_and_args():
    return _harmonic_arg_list + _FENE_arg_list + _Tether_arg_list


def get_bond_args_forces_and_energies():
    harmonic_forces = [-18.9300, -18.2750, -16.6200]
    harmonic_energies = [5.9724, 6.6795, 6.9056]
    FENE_forces = [282.296, 146.288, 88.8238]
    FENE_energies = [70.5638, 49.2476, 35.3135]
    Tether_forces = [0, -0.0244441, -3.57225]
    Tether_energies = [0, 0.000154384, 0.0490934]

    harmonic_args_and_vals = []
    FENE_args_and_vals = []
    Tether_args_and_vals = []
    for i in range(3):
        harmonic_args_and_vals.append(
            (_harmonic_arg_list[i][0], _harmonic_arg_list[i][1],
             harmonic_forces[i], harmonic_energies[i]))
        FENE_args_and_vals.append((_FENE_arg_list[i][0], _FENE_arg_list[i][1],
                                   FENE_forces[i], FENE_energies[i]))
        Tether_args_and_vals.append(
            (_Tether_arg_list[i][0], _Tether_arg_list[i][1], Tether_forces[i],
             Tether_energies[i]))
    return harmonic_args_and_vals + FENE_args_and_vals + Tether_args_and_vals


@pytest.mark.parametrize("bond_cls, potential_kwargs", get_bond_and_args())
def test_before_attaching(bond_cls, potential_kwargs):
    bond_potential = bond_cls()
    bond_potential.params['bond'] = potential_kwargs
    for key in potential_kwargs:
        np.testing.assert_allclose(bond_potential.params['bond'][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("bond_cls, potential_kwargs", get_bond_and_args())
def test_after_attaching(two_particle_snapshot_factory, simulation_factory,
                         bond_cls, potential_kwargs):
    snap = two_particle_snapshot_factory(d=0.969, L=5)
    if snap.communicator.rank == 0:
        snap.bonds.N = 1
        snap.bonds.types = ['bond']
        snap.bonds.typeid[0] = 0
        snap.bonds.group[0] = (0, 1)
    sim = simulation_factory(snap)

    bond_potential = bond_cls()
    bond_potential.params['bond'] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(bond_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         alpha=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in potential_kwargs:
        np.testing.assert_allclose(bond_potential.params['bond'][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("bond_cls, potential_kwargs, force, energy",
                         get_bond_args_forces_and_energies())
def test_forces_and_energies(two_particle_snapshot_factory, simulation_factory,
                             bond_cls, potential_kwargs, force, energy):
    snap = two_particle_snapshot_factory(d=0.969, L=5)
    if snap.communicator.rank == 0:
        snap.bonds.N = 1
        snap.bonds.types = ['bond']
        snap.bonds.typeid[0] = 0
        snap.bonds.group[0] = (0, 1)
        snap.particles.diameter[0] = 0.5
        snap.particles.diameter[1] = 0.5
    sim = simulation_factory(snap)

    bond_potential = bond_cls()
    bond_potential.params['bond'] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(bond_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         alpha=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    sim_energies = sim.operations.integrator.forces[0].energies
    sim_forces = sim.operations.integrator.forces[0].forces
    if sim.device.communicator.rank == 0:
        np.testing.assert_allclose(sum(sim_energies),
                                   energy,
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[0], [force, 0.0, 0.0],
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[1], [-1 * force, 0.0, 0.0],
                                   rtol=1e-2,
                                   atol=1e-5)
