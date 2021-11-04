import numpy as np
import numpy.testing as npt

import hoomd
from hoomd import md


class MyForce(md.force.Custom):

    def __init__(self):
        super().__init__()

    def set_forces(self, timestep):
        with self.cpu_local_force_arrays as arrays:
            arrays.force[:] = -5
            arrays.potential_energy[:] = 37
            arrays.torque[:] = 23
            for i in range(6):
                arrays.virial[:, i] = i


def test_simulation(simulation_factory, two_particle_snapshot_factory):
    """Make sure custom force can plug into simulation without crashing."""
    snap = two_particle_snapshot_factory()
    sim = simulation_factory(snap)
    custom_force = MyForce()
    nvt = md.methods.NPT(hoomd.filter.All(), kT=1, tau=1, S=1, tauS=1,
                         couple="none")
    integrator = md.Integrator(dt=0.005, forces=[custom_force], methods=[nvt])
    sim.operations.integrator = integrator
    sim.run(2)

    npt.assert_allclose(integrator.forces[0].forces, -5)
    npt.assert_allclose(integrator.forces[0].energies, 37)
    npt.assert_allclose(integrator.forces[0].torques, 23)
    for i in range(6):
        npt.assert_allclose(integrator.forces[0].virials[:, i], i)


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
            arrays.force[:] = forces
            arrays.potential_energy[:] = potential


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
    sim.run(100)
    sim2.run(100)

    snap_end = sim.state.get_snapshot()
    snap_end2 = sim2.state.get_snapshot()

    # compare particle properties
    npt.assert_allclose(snap_end.particles.position,
                        snap_end2.particles.position)
    npt.assert_allclose(snap_end.particles.velocity,
                        snap_end2.particles.velocity)
    npt.assert_allclose(integrator.forces[0].forces,
                        integrator2.forces[0].forces)
    npt.assert_allclose(integrator.forces[0].energies,
                        integrator2.forces[0].energies)
    npt.assert_allclose(integrator.forces[0].torques,
                        integrator2.forces[0].torques)
    assert integrator.forces[0].virials == integrator2.forces[0].virials

