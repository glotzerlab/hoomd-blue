import numpy as np

import hoomd
from hoomd import md


class MyConstantForce(md.force.Custom):

    def __init__(self, magnitude):
        super().__init__()
        self._mag = magnitude
        self._direction = np.array([1, 0, 0])

    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as snap, self.cpu_local_force_arrays as arrays:
            rtags = snap.particles.rtag
            position = snap.particles.position[rtags]
            arrays.force[rtags] = self._direction * self._mag
            arrays.potential_energy[rtags] = -self._mag * position[rtags][0]
            arrays.torque[rtags] = np.cross(position, self.forces[rtags])
            arrays.virial[rtags] = ...
