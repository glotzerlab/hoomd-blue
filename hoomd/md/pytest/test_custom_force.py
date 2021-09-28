import numpy as np

import hoomd
from hoomd import md


class MyCustomForce(md.force.Custom):

    def __init__(self, snapshot, filter, magnitude):
        self._snap = snapshot
        self._filt = filter
        self._mag = magnitude

    def set_forces(self, timestep):
        for tag in self._filt:
            position = self._snap.particles.position[tag]
            self.forces[tag] = np.array([1, 0, 0]) * self._mag
            self.energies[tag] = -self._mag * position[tag][0]
            self.torques[tag] = np.cross(position, self.forces[tag])
