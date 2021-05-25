import hoomd
import pytest
import numpy as np

_k = [3.0, 10.0, 5.0]
_t0 = [np.pi / 2, np.pi / 4, np.pi / 6]


def get_args():
    return [{'k': _ki, 't0': _t0i} for _ki, _t0i in zip(_k, _t0)]


@pytest.fixture(scope='session')
def triplet_snapshot_factory(device):
    def make_snapshot(d=1.0, theta_deg=60, particle_types=['A'], dimensions=3, L=20):
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
            s.particles.position[:] = [[-d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.1],
                                       [0.0, 0.0, 0.1],
                                       [d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2), 0.1]]
            s.particles.types = particle_types
            if dimensions == 2:
                box[2] = 0
                s.particles.position[:] = [[-d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2) + 0.1, 0.0],
                                           [0.0, 0.1, 0.0],
                                           [d * np.sin(theta_rad / 2), d * np.cos(theta_rad / 2) + 0.1, 0.0]]
        return s
    return make_snapshot


@pytest.mark.parametrize("argument_dict", get_args())
def test_before_attaching(argument_dict):
    angle_potential = hoomd.md.angle.Cosinesq()
    angle_potential.params['backbone'] = argument_dict
    for key in argument_dict.keys():
        np.testing.assert_allclose(angle_potential.params['backbone'][key],
                                   argument_dict[key],
                                   rtol=1e-6)


@pytest.mark.parametrize("argument_dict", get_args())
def test_after_attaching(triplet_snapshot_factory, simulation_factory, argument_dict):
    snap = triplet_snapshot_factory()
    if snap.exists:
        snap.angles.N = 1
        snap.angles.types = ['backbone']
        snap.angles.typeid[0] = 0
        snap.angles.group[0] = (0, 1, 2)
    sim = simulation_factory(snap)

    angle_potential = hoomd.md.angle.Cosinesq()
    angle_potential.params['backbone'] = argument_dict

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(angle_potential)

    nvt = hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All(), alpha=0.1)
    integrator.methods.append(nvt)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in argument_dict.keys():
        np.testing.assert_allclose(angle_potential.params['backbone'][key],
                                   argument_dict[key],
                                   rtol=1e-6)
