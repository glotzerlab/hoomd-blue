import hoomd
import pytest
import numpy as np

_k = [3.0, 10.0, 5.0]
_d = [-1, 1, 1]
_n = [0.75, 0.5, 0.25]
_phi0 = [np.pi / 2, np.pi / 4, np.pi / 6]


def get_args():
    arg_dicts = []
    for _ki, _di, _ni, _phi0i in zip(_k, _d, _n, _phi0):
        arg_dicts.append({'k': _ki, 'd': _di, 'n': _ni, 'phi0': _phi0i})
    return arg_dicts


def get_args_and_forces_and_energies():
    forces = [-0.9354, 0.9567, 0.2009]
    energies = [0.6666, 9.6194, 4.8673]
    arg_dicts = []
    for _ki, _di, _ni, _phi0i in zip(_k, _d, _n, _phi0):
        arg_dicts.append({'k': _ki, 'd': _di, 'n': _ni, 'phi0': _phi0i})
    return zip(arg_dicts, forces, energies)


@pytest.fixture(scope='session')
def dihedral_snapshot_factory(device):

    def make_snapshot(d=1.0,
                      phi_deg=45,
                      particle_types=['A'],
                      dimensions=3,
                      L=20):
        phi_rad = phi_deg * (np.pi / 180)
        s = hoomd.Snapshot(device.communicator)
        N = 4
        if s.exists:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box
            s.particles.N = N
            # shift particle positions slightly in z so MPI tests pass
            s.particles.position[:] = [
                [0.0, 0.0, 0.1], [d, 0.0, 0.1],
                [0.0, d * np.cos(phi_rad / 2), d * np.sin(phi_rad / 2) + 0.1],
                [d, d * np.cos(phi_rad / 2), -d * np.sin(phi_rad / 2) + 0.1]
            ]

            s.particles.types = particle_types

        return s

    return make_snapshot


@pytest.mark.parametrize("argument_dict", get_args())
def test_before_attaching(argument_dict):
    dihedral_potential = hoomd.md.dihedral.Harmonic()
    dihedral_potential.params['backbone'] = argument_dict
    for key in argument_dict.keys():
        np.testing.assert_allclose(dihedral_potential.params['backbone'][key],
                                   argument_dict[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("argument_dict", get_args())
def test_after_attaching(dihedral_snapshot_factory, simulation_factory,
                         argument_dict):
    snap = dihedral_snapshot_factory()
    if snap.exists:
        snap.dihedrals.N = 1
        snap.dihedrals.types = ['backbone']
        snap.dihedrals.typeid[0] = 0
        snap.dihedrals.group[0] = (0, 1, 2, 3)
    sim = simulation_factory(snap)

    dihedral_potential = hoomd.md.dihedral.Harmonic()
    dihedral_potential.params['backbone'] = argument_dict

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(dihedral_potential)

    nvt = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All(), alpha=0.1)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in argument_dict.keys():
        np.testing.assert_allclose(dihedral_potential.params['backbone'][key],
                                   argument_dict[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("args_and_force_and_energy",
                         get_args_and_forces_and_energies())
def test_forces_and_energies(dihedral_snapshot_factory, simulation_factory,
                             args_and_force_and_energy):
    phi_deg = 45
    phi_rad = phi_deg * (np.pi / 180)
    snap = dihedral_snapshot_factory(phi_deg=phi_deg)
    if snap.exists:
        snap.dihedrals.N = 1
        snap.dihedrals.types = ['backbone']
        snap.dihedrals.typeid[0] = 0
        snap.dihedrals.group[0] = (0, 1, 2, 3)
    sim = simulation_factory(snap)

    argument_dict, force, energy = args_and_force_and_energy
    force_array = force * np.asarray(
        [np.cos(phi_rad / 2), np.sin(phi_rad / 2), 0])
    dihedral_potential = hoomd.md.dihedral.Harmonic()
    dihedral_potential.params['backbone'] = argument_dict

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(dihedral_potential)

    nvt = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All(), alpha=0.1)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)

    sim_energies = sim.operations.integrator.forces[0].energy
    sim_forces = sim.operations.integrator.forces[0].forces
    print(sim_energies)
    if sim.device.communicator.rank == 0:
        np.testing.assert_allclose(sim_energies, energy, rtol=1e-2, atol=1e-5)
        np.testing.assert_allclose(sim_forces[0],
                                   force_array,
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[1], [0, -1 * force, 0],
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(
            sim_forces[2],
            [-1 * force_array[0], force_array[1], force_array[2]],
            rtol=1e-2,
            atol=1e-5)
