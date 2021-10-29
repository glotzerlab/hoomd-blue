import numpy as np
import numpy.testing as npt

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
            arrays.potential_energy[rtags] = -self._mag * position[rtags][:, 0]
            arrays.torque[rtags] = np.cross(position, arrays.force[rtags])

            # set the virial stress coefficients
            arrays.virial[rtags][:, 0] = self._force[0] * position[rtags][:, 0]
            arrays.virial[rtags][:, 1] = self._force[0] * position[rtags][:, 1]
            arrays.virial[rtags][:, 2] = self._force[0] * position[rtags][:, 2]
            arrays.virial[rtags][:, 3] = self._force[1] * position[rtags][:, 0]
            arrays.virial[rtags][:, 4] = self._force[1] * position[rtags][:, 1]
            arrays.virial[rtags][:, 5] = self._force[2] * position[rtags][:, 2]


def test_simulation(simulation_factory, two_particle_snapshot_factory):
    """Make sure custom force can plug into simulation without crashing."""
    snap = two_particle_snapshot_factory()
    sim = simulation_factory(snap)
    custom_grav = MyConstantForce(2)
    nvt = md.methods.NVT(hoomd.filter.All(), kT=1, tau=1)
    integrator = md.Integrator(dt=0.005, forces=[custom_grav], methods=[nvt])
    sim.operations.integrator = integrator
    sim.run(2)


class MyPeriodicField(md.force.Custom):

    def __init__(self, A, i, p, w):
        super().__init__()
        self._A = A
        self._i = i
        self._p = p
        self._w = w

    def _evaluate_periodic(self, snapshot):
        """Evaluate force and energy in python."""
        box = snapshot.global_box
        positions = snapshot.particles.position
        a1, a2, a3 = box.lattice_vectors
        V = np.dot(a1, np.cross(a2, a3))
        b1 = 2 * np.pi / V * np.cross(a2, a3)
        b2 = 2 * np.pi / V * np.cross(a3, a1)
        b3 = 2 * np.pi / V * np.cross(a1, a2)
        b = {0: b1, 1: b2, 2: b3}.get(self._i)
        energies = self._A * np.tanh(
            1 / (2 * np.pi * self._p * self._w) * np.cos(self._p * np.dot(positions, b)))
        forces = self._A / (2 * np.pi * self._w) * np.sin(self._p * np.dot(positions, b))
        forces *= 1 - (np.tanh(
            np.cos(self._p * np.dot(positions, b)) / (2 * np.pi * self._p * self._w)))**2
        forces = np.outer(forces, b)
        return forces, energies

    def set_forces(self, timestep):
        with self._state.cpu_local_snapshot as snap, \
                self.cpu_local_force_arrays as arrays:
            forces, potential = self._evaluate_periodic(snap)

            position = snap.particles.position
            #print(position)
            print(arrays.force)
            arrays.force = forces
            print(arrays.force)
            arrays.potential_energy = potential
            arrays.torque = np.cross(position, forces)
            arrays.virial[:, 0] = forces[:, 0] * position[:, 0]
            arrays.virial[:, 1] = forces[:, 0] * position[:, 1]
            arrays.virial[:, 2] = forces[:, 0] * position[:, 2]
            arrays.virial[:, 3] = forces[:, 1] * position[:, 0]
            arrays.virial[:, 4] = forces[:, 1] * position[:, 1]
            arrays.virial[:, 5] = forces[:, 2] * position[:, 2]


def test_compare_to_periodic(simulation_factory, two_particle_snapshot_factory):
    """Test hoomd external periodic compared to a python version."""
    # sim with built-in force field
    snap = two_particle_snapshot_factory()
    sim = simulation_factory(snap)
    periodic = md.external.field.Periodic()
    periodic.params['A'] = dict(A=1, i=0, p=1, w=1)
    nvt = md.methods.NVT(hoomd.filter.All(), kT=1, tau=1)
    integrator = md.Integrator(dt=0.005, forces=[periodic], methods=[nvt])
    sim.operations.integrator = integrator

    # sim with custom but equivalent force field
    snap2 = two_particle_snapshot_factory()
    sim2 = simulation_factory(snap)
    periodic2 = MyPeriodicField(A=1, i=0, p=1, w=1)
    nvt2 = md.methods.NVT(hoomd.filter.All(), kT=1, tau=1)
    integrator2 = md.Integrator(dt=0.005, forces=[periodic2], methods=[nvt2])
    sim2.operations.integrator = integrator2

    # run simulations next to each other
    for i in range(100):
        print(i)
        #sim.run(1)
        #print(sim.state.get_snapshot().particles.position)
        #print(sim.operations.integrator.forces[0].forces)
        sim2.run(1)
    snap_end = sim.state.get_snapshot()
    snap_end2 = sim2.state.get_snapshot()

    # compare particle properties
    npt.assert_allclose(snap_end.particles.position,
                        snap_end2.particles.position)
    #print(integrator.forces[0].forces)
    #print(integrator2.forces[0].forces)
    #npt.assert_allclose(integrator.forces[0].forces,
    #                    integrator2.forces[0].forces)
    #npt.assert_allclose(integrator.forces[0].torques,
    #                    integrator2.forces[0].torques)
    #print(integrator.forces[0].virials)
    #print(integrator2.forces[0].virials)
    npt.assert_allclose(integrator.forces[0].virials,
                        integrator2.forces[0].virials)
    #npt.assert_allclose(integrator.forces[0].energies,
    #                    integrator2.forces[0].energies)

