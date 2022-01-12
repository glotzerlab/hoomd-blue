# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy as np

_harmonic_args = {
    'k': [3.0, 10.0, 5.0],
    't0': [np.pi / 2, np.pi / 4, np.pi / 6]
}
_harmonic_arg_list = [(hoomd.md.angle.Harmonic, dict(zip(_harmonic_args, val)))
                      for val in zip(*_harmonic_args.values())]

_cosinesq_args = {
    'k': [3.0, 10.0, 5.0],
    't0': [np.pi / 2, np.pi / 4, np.pi / 6]
}
_cosinesq_arg_list = [(hoomd.md.angle.Cosinesq, dict(zip(_cosinesq_args, val)))
                      for val in zip(*_cosinesq_args.values())]


def get_angle_and_args():
    return _harmonic_arg_list + _cosinesq_arg_list


def get_angle_args_forces_and_energies():
    harmonic_forces = [-1.5708, 2.6180, 2.6180]
    harmonic_energies = [0.4112, 0.3427, 0.6854]
    cosinesq_forces = [-1.29904, 1.7936, 1.58494]
    cosinesq_energies = [0.375, 0.214466, 0.334936]

    harmonic_args_and_vals = []
    cosinesq_args_and_vals = []
    for i in range(3):
        harmonic_args_and_vals.append(
            (_harmonic_arg_list[i][0], _harmonic_arg_list[i][1],
             harmonic_forces[i], harmonic_energies[i]))
        cosinesq_args_and_vals.append(
            (_cosinesq_arg_list[i][0], _cosinesq_arg_list[i][1],
             cosinesq_forces[i], cosinesq_energies[i]))

    return harmonic_args_and_vals + cosinesq_args_and_vals


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
        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box
            s.particles.N = N

            base_positions = np.array(
                [[-d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.0],
                 [0.0, 0.0, 0.0],
                 [d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.0]])
            # move particles slightly in direction of MPI decomposition which
            # varies by simulation dimension
            nudge_dimension = 2 if dimensions == 3 else 1
            base_positions[:, nudge_dimension] += 0.1
            s.particles.position[:] = base_positions
            s.particles.types = particle_types
            s.angles.N = 1
            s.angles.types = ['backbone']
            s.angles.typeid[0] = 0
            s.angles.group[0] = (0, 1, 2)
        return s

    return make_snapshot


@pytest.mark.parametrize("angle_cls, potential_kwargs", get_angle_and_args())
def test_before_attaching(angle_cls, potential_kwargs):
    angle_potential = angle_cls()
    angle_potential.params['backbone'] = potential_kwargs
    for key in potential_kwargs:
        np.testing.assert_allclose(angle_potential.params['backbone'][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("angle_cls, potential_kwargs", get_angle_and_args())
def test_after_attaching(triplet_snapshot_factory, simulation_factory,
                         angle_cls, potential_kwargs):
    snap = triplet_snapshot_factory()
    sim = simulation_factory(snap)

    angle_potential = angle_cls()
    angle_potential.params['backbone'] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(angle_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         alpha=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in potential_kwargs:
        np.testing.assert_allclose(angle_potential.params['backbone'][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("angle_cls, potential_kwargs, force, energy",
                         get_angle_args_forces_and_energies())
def test_forces_and_energies(triplet_snapshot_factory, simulation_factory,
                             angle_cls, potential_kwargs, force, energy):
    theta_deg = 60
    theta_rad = theta_deg * (np.pi / 180)
    snap = triplet_snapshot_factory(theta_deg=theta_deg)
    sim = simulation_factory(snap)

    force_array = force * np.asarray(
        [np.cos(theta_rad / 2), np.sin(theta_rad / 2), 0])
    angle_potential = angle_cls()
    angle_potential.params['backbone'] = potential_kwargs

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
