import hoomd
import pytest
import numpy as np

_harmonic_args = {
    'k': [3.0, 10.0, 5.0],
    't0': [np.pi / 2, np.pi / 4, np.pi / 6]
}

_cosinesq_args = {
    'k': [3.0, 10.0, 5.0],
    't0': [np.pi / 2, np.pi / 4, np.pi / 6]
}


def get_angle_and_args():
    harmonic_arg_list = [
        dict(zip(_harmonic_args, val)) for val in zip(*_harmonic_args.values())
    ]
    cosinesq_arg_list = [
        dict(zip(_cosinesq_args, val)) for val in zip(*_cosinesq_args.values())
    ]
    angle_and_args = []
    for args in harmonic_arg_list:
        angle_and_args.append((hoomd.md.angle.Harmonic, args))
    for args in cosinesq_arg_list:
        angle_and_args.append((hoomd.md.angle.Cosinesq, args))
    return angle_and_args


def get_angle_args_forces_and_energies():
    harmonic_arg_list = [
        dict(zip(_harmonic_args, val)) for val in zip(*_harmonic_args.values())
    ]
    cosinesq_arg_list = [
        dict(zip(_cosinesq_args, val)) for val in zip(*_cosinesq_args.values())
    ]
    harmonic_forces = [-1.5708, 2.6180, 2.6180]
    harmonic_energies = [0.4112, 0.3427, 0.6854]
    cosinesq_forces = [-1.29904, 1.7936, 1.58494]
    cosinesq_energies = [0.375, 0.214466, 0.334936]

    angle_args_forces_and_energies = []
    for i in range(3):
        angle_args_forces_and_energies.append(
            (hoomd.md.angle.Harmonic, harmonic_arg_list[i], harmonic_forces[i],
             harmonic_energies[i]))
    for i in range(3):
        angle_args_forces_and_energies.append(
            (hoomd.md.angle.Cosinesq, cosinesq_arg_list[i], cosinesq_forces[i],
             cosinesq_energies[i]))
    return angle_args_forces_and_energies


@pytest.fixture(scope='session')
def triplet_snapshot_factory(device):

    def make_snapshot(d=1.0,
                      theta_deg=60,
                      particle_types=['A'],
                      dimensions=3,
                      L=20):
        theta_rad = theta_deg * (np.pi / 180)
        s = hoomd.Snapshot(device.communicator)
        N = 3
        if s.exists:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box
            s.particles.N = N
            # shift particle positions slightly in z so MPI tests pass
            s.particles.position[:] = [
                [-d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.1],
                [0.0, 0.0, 0.1],
                [d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.1]
            ]
            s.particles.types = particle_types
            if dimensions == 2:
                box[2] = 0
                s.particles.position[:] = [[
                    -d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2) + 0.1,
                    0.0
                ], [0.0, 0.1, 0.0],
                                           [
                                               d * np.sin(theta_rad / 2),
                                               d * np.cos(theta_rad / 2) + 0.1,
                                               0.0
                                           ]]
        return s

    return make_snapshot


@pytest.mark.parametrize("angle_and_args", get_angle_and_args())
def test_before_attaching(angle_and_args):
    angle, args = angle_and_args
    angle_potential = angle()
    angle_potential.params['backbone'] = args
    for key in args.keys():
        np.testing.assert_allclose(angle_potential.params['backbone'][key],
                                   args[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("angle_and_args", get_angle_and_args())
def test_after_attaching(triplet_snapshot_factory, simulation_factory,
                         angle_and_args):
    snap = triplet_snapshot_factory()
    if snap.exists:
        snap.angles.N = 1
        snap.angles.types = ['backbone']
        snap.angles.typeid[0] = 0
        snap.angles.group[0] = (0, 1, 2)
    sim = simulation_factory(snap)

    angle, args = angle_and_args
    angle_potential = angle()
    angle_potential.params['backbone'] = args

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(angle_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         alpha=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in args.keys():
        np.testing.assert_allclose(angle_potential.params['backbone'][key],
                                   args[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("angle_args_force_and_energy",
                         get_angle_args_forces_and_energies())
def test_forces_and_energies(triplet_snapshot_factory, simulation_factory,
                         angle_args_force_and_energy):
    theta_deg = 60
    theta_rad = theta_deg * (np.pi / 180)
    snap = triplet_snapshot_factory(theta_deg=theta_deg)
    if snap.exists:
        snap.angles.N = 1
        snap.angles.types = ['backbone']
        snap.angles.typeid[0] = 0
        snap.angles.group[0] = (0, 1, 2)
    sim = simulation_factory(snap)

    angle, args, force, energy = angle_args_force_and_energy
    force_array = force * np.asarray(
        [np.cos(theta_rad / 2), np.sin(theta_rad / 2), 0])
    angle_potential = angle()
    angle_potential.params['backbone'] = args

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(angle_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         alpha=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    sim_energies = sim.operations.integrator.forces[0].energy
    sim_forces = sim.operations.integrator.forces[0].forces
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
