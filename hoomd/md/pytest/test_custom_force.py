import numpy as np

import hoomd
from hoomd import md


class MyConstantForce(md.force.Custom):

    def __init__(self, magnitude):
        super().__init__()
        self._mag = magnitude
        self._direction = np.array([1, 0, 0])
        self._force = self._direction * self._mag

    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as snap, self.cpu_local_force_arrays as arrays:
            rtags = snap.particles.rtag
            position = snap.particles.position[rtags]
            arrays.force[rtags] = self._force[None, :]
            arrays.potential_energy[rtags] = -self._mag * position[rtags][0]
            arrays.torque[rtags] = np.cross(position, arrays.force[rtags])

            # set the virial stress coefficients
            arrays.virial[rtags][:, 0] = force[0] * position[:, 0]
            arrays.virial[rtags][:, 1] = force[0] * position[:, 1]
            arrays.virial[rtags][:, 2] = force[0] * position[:, 2]
            arrays.virial[rtags][:, 3] = force[1] * position[:, 0]
            arrays.virial[rtags][:, 4] = force[1] * position[:, 1]
            arrays.virial[rtags][:, 5] = force[2] * position[:, 2]


def test_simulation(simulation_factory, two_particle_snapshot_factory):
    snap = two_particle_snapshot_factory()
    sim = simulation_factory(snap)
    custom_grav = MyConstantForce(2)
    integrator = hoomd.md.Integrator(dt=0.005, forces=[custom_grav])
    sim.operations.integrator = integrator

    sim.run(2)
