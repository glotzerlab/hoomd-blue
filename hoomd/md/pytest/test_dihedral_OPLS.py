import hoomd
import pytest
import numpy as np

_k1 = [1.0, 0.5, 2.0]
_k2 = [1.5, 2.5, 1.0]
_k3 = [0.5, 1.5, 0.25]
_k4 = [0.75, 1.0, 3.5]


def get_args():
    arg_dicts = []
    for _k1i, _k2i, _k3i, _k4i in zip(_k1, _k2, _k3, _k4):
        arg_dicts.append({'k1': _k1i, 'k2': _k2i, 'k3': _k3i, 'k4': _k4i})
    return arg_dicts


def get_args_and_forces_and_energies():
    forces = [0.616117, 0.732233, 0.0277282]
    energies = [2.42678, 2.89645, 5.74372]
    arg_dicts = []
    for _k1i, _k2i, _k3i, _k4i in zip(_k1, _k2, _k3, _k4):
        arg_dicts.append({'k1': _k1i, 'k2': _k2i, 'k3': _k3i, 'k4': _k4i})
    return zip(arg_dicts, forces, energies)


@pytest.fixture(scope='session')
def dihedral_snapshot_factory(device):
    def make_snapshot(d=1.0, phi_deg=45, particle_types=['A'], dimensions=3, L=20):
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
            s.particles.position[:] = [[-d * np.sin(phi_rad), -d * np.cos(phi_rad), 0.1],
                                       [-d / 2, 0.0, 0.1],
                                       [d / 2, 0.0, 0.1],
                                       [d * np.sin(phi_rad), d * np.cos(phi_rad), 0.1]]
            s.particles.types = particle_types
            if dimensions == 2:
                box[2] = 0
                s.particles.position[:] = [[-d * np.sin(phi_rad), -d * np.cos(phi_rad) + 0.1, 0.0],
                                           [-d / 2, 0.1, 0.0],
                                           [d / 2, 0.1, 0.0],
                                           [d * np.sin(phi_rad), d * np.cos(phi_rad) + 0.1, 0.0]]
        return s
    return make_snapshot


@pytest.mark.parametrize("argument_dict", get_args())
def test_before_attaching(argument_dict):
    dihedral_potential = hoomd.md.dihedral.OPLS()
    dihedral_potential.params['backbone'] = argument_dict
    for key in argument_dict.keys():
        np.testing.assert_allclose(dihedral_potential.params['backbone'][key],
                                   argument_dict[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("argument_dict", get_args())
def test_after_attaching(dihedral_snapshot_factory, simulation_factory, argument_dict):
    snap = dihedral_snapshot_factory()
    if snap.exists:
        snap.dihedrals.N = 1
        snap.dihedrals.types = ['backbone']
        snap.dihedrals.typeid[0] = 0
        snap.dihedrals.group[0] = (0, 1, 2, 3)
    sim = simulation_factory(snap)

    dihedral_potential = hoomd.md.dihedral.OPLS()
    dihedral_potential.params['backbone'] = argument_dict

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(dihedral_potential)

    nvt = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All(), alpha=0.1)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in argument_dict.keys():
        print(key)
        print(dihedral_potential.params['backbone'][key])
        print(argument_dict[key])
        print()
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
    force_array = force * np.asarray([np.cos(phi_rad / 2), np.sin(phi_rad / 2), 0])
    dihedral_potential = hoomd.md.dihedral.OPLS()
    dihedral_potential.params['backbone'] = argument_dict

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(dihedral_potential)

    langevin = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All(), alpha=0.1)
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
                                   [0, -1 * force, 0],
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[2],
                                   [-1 * force_array[0], force_array[1], force_array[2]],
                                   rtol=1e-2,
                                   atol=1e-5)
