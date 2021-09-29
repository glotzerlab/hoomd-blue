import numpy as np

import hoomd
from hoomd import md


class MyCustomForce(md.force.Custom):

    def __init__(self, filter, magnitude):
        super().__init__()  # if we could make this work without this, that'd be great
        self._filt = filter
        self._mag = magnitude

    def set_forces(self, timestep):
        tags = self._filt(self._state)
        with self._state.cpu_local_snapshot as snap, self.cpu_local_force_arrays as arrays:
            for tag in tags:
                position = snap.particles.position[tag]
                arrays.force[tag] = np.array([1, 0, 0]) * self._mag
                arrays.potential_energy[tag] = -self._mag * position[tag][0]
                arrays.torque[tag] = np.cross(position, self.forces[tag])
                arrays.virial[tag] = ...
