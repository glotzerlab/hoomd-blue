import hoomd
import pytest
import numpy as np

_harmonic_args = {
    'k': [3.0, 10.0, 5.0],
    'd': [-1, 1, 1],
    'n': [0.75, 0.5, 0.25],
    'phi0': [np.pi / 2, np.pi / 4, np.pi / 6]
}

_OPLS_args = {
    'k1': [1.0, 0.5, 2.0],
    'k2': [1.5, 2.5, 1.0],
    'k3': [0.5, 1.5, 0.25],
    'k4': [0.75, 1.0, 3.5]
}


def get_dihedral_and_args():
    harmonic_arg_list = [
        dict(zip(_harmonic_args, val)) for val in zip(*_harmonic_args.values())
    ]
    OPLS_arg_list = [
        dict(zip(_OPLS_args, val)) for val in zip(*_OPLS_args.values())
    ]
    dihedral_and_args = []
    for args in harmonic_arg_list:
        dihedral_and_args.append((hoomd.md.dihedral.Harmonic, args))
    for args in OPLS_arg_list:
        dihedral_and_args.append((hoomd.md.dihedral.OPLS, args))
    return dihedral_and_args


def get_dihedral_args_forces_and_energies():
    harmonic_arg_list = [
        dict(zip(_harmonic_args, val)) for val in zip(*_harmonic_args.values())
    ]
    OPLS_arg_list = [
        dict(zip(_OPLS_args, val)) for val in zip(*_OPLS_args.values())
    ]
    harmonic_forces = [-0.9354, 0.9567, 0.2009]
    harmonic_energies = [0.6666, 9.6194, 4.8673]
    OPLS_forces = [0.616117, 0.732233, 0.0277282]
    OPLS_energies = [2.42678, 2.89645, 5.74372]

    dihedral_args_forces_and_energies = []
    for i in range(3):
        dihedral_args_forces_and_energies.append(
            (hoomd.md.dihedral.Harmonic, harmonic_arg_list[i],
             harmonic_forces[i], harmonic_energies[i]))
    for i in range(3):
        dihedral_args_forces_and_energies.append(
            (hoomd.md.dihedral.OPLS, OPLS_arg_list[i], OPLS_forces[i],
             OPLS_energies[i]))
    return dihedral_args_forces_and_energies


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


@pytest.mark.parametrize("dihedral_and_args", get_dihedral_and_args())
def test_before_attaching(dihedral_and_args):
    dihedral, args = dihedral_and_args
    dihedral_potential = dihedral()
    dihedral_potential.params['dihedral'] = args
    for key in args.keys():
        np.testing.assert_allclose(dihedral_potential.params['dihedral'][key],
                                   args[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("dihedral_and_args", get_dihedral_and_args())
def test_after_attaching(two_particle_snapshot_factory, simulation_factory,
                         dihedral_and_args):
    snap = two_particle_snapshot_factory(d=0.969, L=5)
    if snap.exists:
        snap.dihedrals.N = 1
        snap.dihedrals.types = ['dihedral']
        snap.dihedrals.typeid[0] = 0
        snap.dihedrals.group[0] = (0, 1, 2, 3)
    sim = simulation_factory(snap)

    dihedral, args = dihedral_and_args
    dihedral_potential = dihedral()
    dihedral_potential.params['dihedral'] = args

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(dihedral_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         alpha=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in args.keys():
        np.testing.assert_allclose(dihedral_potential.params['dihedral'][key],
                                   args[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("dihedral_args_force_and_energy",
                         get_dihedral_args_forces_and_energies())
def test_forces_and_energies(two_particle_snapshot_factory, simulation_factory,
                             dihedral_args_force_and_energy):
    snap = two_particle_snapshot_factory(d=0.969, L=5)
    if snap.exists:
        snap.dihedrals.N = 1
        snap.dihedrals.types = ['dihedral']
        snap.dihedrals.typeid[0] = 0
        snap.dihedrals.group[0] = (0, 1)
        snap.particles.diameter[0] = 0.5
        snap.particles.diameter[1] = 0.5
    sim = simulation_factory(snap)

    dihedral, args, force, energy = dihedral_args_force_and_energy
    dihedral_potential = dihedral()
    dihedral_potential.params['dihedral'] = args

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
        np.testing.assert_allclose(sim_forces[0], [force, 0.0, 0.0],
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces[1], [-1 * force, 0.0, 0.0],
                                   rtol=1e-2,
                                   atol=1e-5)
