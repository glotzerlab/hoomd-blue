import pytest
import numpy as np
import numpy.testing as npt

try:
    import cupy
    CUPY_IMPORTED = True
except:
    CUPY_IMPORTED = False

import hoomd
from hoomd import md


class MyForceCPU(md.force.Custom):

    def __init__(self):
        super().__init__()

    def set_forces(self, timestep):
        with self.cpu_local_force_arrays as arrays:
            arrays.force[:] = -5
            arrays.potential_energy[:] = 37
            arrays.torque[:] = 23
            for i in range(6):
                arrays.virial[:, i] = i


class MyForceGPU(md.force.Custom):

    def __init__(self):
        super().__init__()

    def set_forces(self, timestep):
        with self.gpu_local_force_arrays as arrays:
            arrays.force[:] = -5
            arrays.potential_energy[:] = 37
            arrays.torque[:] = 23
            # cupy won't let me use two indices on arrays of length 0
            if len(arrays.virial) > 0:
                for i in range(6):
                    arrays.virial[:, i] = i


def _skip_if_gpu_with_cpu_device(force_cls, sim):
    gpu_class = force_cls.__name__.endswith("GPU")
    cpu_device = isinstance(sim.device, hoomd.device.CPU)
    if cpu_device and gpu_class:
        pytest.skip("Cannot access gpu force arrays with CPU device")


def _skip_if_cupy_not_imported_and_gpu_device(sim):
    gpu_device = isinstance(sim.device, hoomd.device.GPU)
    if gpu_device and not CUPY_IMPORTED:
        pytest.skip("Cannot run this test on GPU without cupy")


@pytest.mark.parametrize("force_cls", [MyForceCPU, MyForceGPU], ids=lambda x: x.__name__)
def test_simulation(force_cls, simulation_factory, two_particle_snapshot_factory):
    """Make sure custom force can plug into simulation without crashing."""
    snap = two_particle_snapshot_factory()
    sim = simulation_factory(snap)
    _skip_if_gpu_with_cpu_device(force_cls, sim)
    custom_force = force_cls()
    nvt = md.methods.NPT(hoomd.filter.All(), kT=1, tau=1, S=1, tauS=1,
                         couple="none")
    integrator = md.Integrator(dt=0.005, forces=[custom_force], methods=[nvt])
    sim.operations.integrator = integrator
    sim.run(2)

    force_arr = integrator.forces[0].forces
    energy_arr = integrator.forces[0].energies
    torque_arr = integrator.forces[0].torques
    virial_arr = integrator.forces[0].virials
    if sim.device.communicator.rank == 0:
        npt.assert_allclose(force_arr, -5)
        npt.assert_allclose(energy_arr, 37)
        npt.assert_allclose(torque_arr, 23)
        for i in range(6):
            npt.assert_allclose(virial_arr[:, i], i)


class MyPeriodicFieldCPU(md.force.Custom):

    def __init__(self, A, i, p, w):
        super().__init__()
        self._A = A
        self._i = i
        self._p = p
        self._w = w

    def _numpy_array(self, arr):
        """If arr is hoomd array change it to numpy."""
        if arr.__class__.__name__ == 'HOOMDGPUArray':
            return arr.get()
        else:
            return arr

    def _evaluate_periodic(self, snapshot):
        """Evaluate force and energy in python."""
        box = snapshot.global_box
        positions = self._numpy_array(snapshot.particles.position)

        # if no particles on this rank, return
        if positions.shape == (0,):
            return np.array([]), np.array([])

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


class MyPeriodicFieldGPU(MyPeriodicFieldCPU):

    def __init__(self, A, i, p, w):
        super().__init__(A, i, p, w)

    def set_forces(self, timestep):
        with self._state.gpu_local_snapshot as snap, \
                self.gpu_local_force_arrays as arrays:
            forces, potential = self._evaluate_periodic(snap)
            arrays.force[:] = cupy.array(forces)
            arrays.potential_energy[:] = cupy.array(potential)



@pytest.mark.parametrize("force_cls", [MyPeriodicFieldCPU, MyPeriodicFieldGPU], ids=lambda x: x.__name__)
def test_compare_to_periodic(force_cls, simulation_factory, two_particle_snapshot_factory):
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
    _skip_if_gpu_with_cpu_device(force_cls, sim2)
    _skip_if_cupy_not_imported_and_gpu_device(sim2)
    periodic2 = force_cls(A=1, i=0, p=1, w=1)
    nvt2 = md.methods.NVT(hoomd.filter.All(), kT=1, tau=1)
    integrator2 = md.Integrator(dt=0.005, forces=[periodic2], methods=[nvt2])
    sim2.operations.integrator = integrator2

    # run simulations next to each other
    sim.run(100)
    sim2.run(100)

    snap_end = sim.state.get_snapshot()
    snap_end2 = sim2.state.get_snapshot()

    # compare particle properties
    if sim.device.communicator.rank == 0:
        positions1 = snap_end.particles.position
        positions2 = snap_end2.particles.position
        npt.assert_allclose(positions1, positions2)

        velocities1 = snap_end.particles.velocity
        velocities2 = snap_end2.particles.velocity
        npt.assert_allclose(velocities1, velocities2)

    forces1 = integrator.forces[0].forces
    forces2 = integrator2.forces[0].forces
    if sim.device.communicator.rank == 0:
        npt.assert_allclose(forces1, forces2)

    energies1 = integrator.forces[0].energies
    energies2 = integrator2.forces[0].energies
    if sim.device.communicator.rank == 0:
        npt.assert_allclose(energies1, energies2)

    torques1 = integrator.forces[0].torques
    torques2 = integrator2.forces[0].torques
    if sim.device.communicator.rank == 0:
        npt.assert_allclose(torques1, torques2)

    assert integrator.forces[0].virials == integrator2.forces[0].virials

