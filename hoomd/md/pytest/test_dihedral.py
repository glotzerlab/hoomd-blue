# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy as np

_harmonic_args = {
    'k': [3.0, 10.0, 5.0],
    'd': [-1, 1, 1],
    'n': [2, 1, 3],
    'phi0': [np.pi / 2, np.pi / 4, np.pi / 6]
}
_harmonic_arg_list = [(hoomd.md.dihedral.Harmonic,
                       dict(zip(_harmonic_args, val)))
                      for val in zip(*_harmonic_args.values())]

_OPLS_args = {
    'k1': [1.0, 0.5, 2.0],
    'k2': [1.5, 2.5, 1.0],
    'k3': [0.5, 1.5, 0.25],
    'k4': [0.75, 1.0, 3.5]
}
_OPLS_arg_list = [(hoomd.md.dihedral.OPLS, dict(zip(_OPLS_args, val)))
                  for val in zip(*_OPLS_args.values())]


def get_dihedral_and_args():
    return _harmonic_arg_list + _OPLS_arg_list


def get_dihedral_args_forces_and_energies():
    harmonic_forces = [0.0, 5.0, 1.9411]
    harmonic_energies = [3.0, 5.0, 0.0852]
    OPLS_forces = [-0.616117, -0.732233, -0.0277282]
    OPLS_energies = [2.42678, 2.89645, 5.74372]

    harmonic_args_and_vals = []
    OPLS_args_and_vals = []
    for i in range(3):
        harmonic_args_and_vals.append(
            (_harmonic_arg_list[i][0], _harmonic_arg_list[i][1],
             harmonic_forces[i], harmonic_energies[i]))
        OPLS_args_and_vals.append((_OPLS_arg_list[i][0], _OPLS_arg_list[i][1],
                                   OPLS_forces[i], OPLS_energies[i]))

    return harmonic_args_and_vals + OPLS_args_and_vals


@pytest.fixture(scope='session')
def dihedral_snapshot_factory(device):

    def make_snapshot(d=1.0, phi_deg=45, particle_types=['A'], L=20):
        phi_rad = phi_deg * (np.pi / 180)
        # the central particles are along the x-axis, so phi is determined from
        # the angle in the yz plane.

        s = hoomd.Snapshot(device.communicator)
        N = 4
        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            s.configuration.box = box
            s.particles.N = N
            s.particles.types = particle_types
            # shift particle positions slightly in z so MPI tests pass
            s.particles.position[:] = [
                [0.0, d * np.cos(phi_rad / 2), d * np.sin(phi_rad / 2) + 0.1],
                [0.0, 0.0, 0.1], [d, 0.0, 0.1],
                [d, d * np.cos(phi_rad / 2), -d * np.sin(phi_rad / 2) + 0.1]
            ]

            s.dihedrals.N = 1
            s.dihedrals.types = ['dihedral']
            s.dihedrals.typeid[0] = 0
            s.dihedrals.group[0] = (0, 1, 2, 3)

        return s

    return make_snapshot


@pytest.mark.parametrize("dihedral_cls, potential_kwargs",
                         get_dihedral_and_args())
def test_before_attaching(dihedral_cls, potential_kwargs):
    dihedral_potential = dihedral_cls()
    dihedral_potential.params['dihedral'] = potential_kwargs
    for key in potential_kwargs:
        np.testing.assert_allclose(dihedral_potential.params['dihedral'][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("dihedral_cls, potential_kwargs",
                         get_dihedral_and_args())
def test_after_attaching(dihedral_snapshot_factory, simulation_factory,
                         dihedral_cls, potential_kwargs):
    snap = dihedral_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snap)

    dihedral_potential = dihedral_cls()
    dihedral_potential.params['dihedral'] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(dihedral_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         alpha=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in potential_kwargs:
        np.testing.assert_allclose(dihedral_potential.params['dihedral'][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("dihedral_cls, potential_kwargs, force, energy",
                         get_dihedral_args_forces_and_energies())
def test_forces_and_energies(dihedral_snapshot_factory, simulation_factory,
                             dihedral_cls, potential_kwargs, force, energy):
    phi_deg = 45
    phi_rad = phi_deg * (np.pi / 180)
    snap = dihedral_snapshot_factory(phi_deg=phi_deg)
    sim = simulation_factory(snap)

    # the dihedral angle is in yz plane, thus no force along x axis
    force_array = force * np.asarray(
        [0, np.sin(-phi_rad / 2), np.cos(-phi_rad / 2)])
    dihedral_potential = dihedral_cls()
    dihedral_potential.params['dihedral'] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(dihedral_potential)

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
        np.testing.assert_allclose(sim_forces[0],
                                   force_array,
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[1],
                                   -1 * force_array,
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[2],
                                   [0, -1 * force_array[1], force_array[2]],
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[3],
                                   [0, force_array[1], -1 * force_array[2]],
                                   rtol=1e-2,
                                   atol=1e-5)
