import pytest
import numpy as np
import numpy.testing as npt

try:
    import cupy
    CUPY_IMPORTED = True
except ImportError:
    CUPY_IMPORTED = False

# mpi4py is needed for the ghost data test
try:
    from mpi4py import MPI
    MPI4PY_IMPORTED = True
except ImportError:
    MPI4PY_IMPORTED = False

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
            arrays.virial[:] = np.arange(6)[None, :]


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


def _skip_if_cupy_not_imported_and_gpu_device(sim):
    """Force classes cant set values if cupy isn't installed."""
    gpu_device = isinstance(sim.device, hoomd.device.GPU)
    if gpu_device and not CUPY_IMPORTED:
        pytest.skip("Cannot run this test on GPU without cupy")


def _using_gpu_with_cpu_device(force_cls, sim):
    """Sims should throw an error if this returns true."""
    gpu_class = force_cls.__name__.endswith("GPU")
    cpu_device = isinstance(sim.device, hoomd.device.CPU)
    return cpu_device and gpu_class


def _try_running_sim(sim, tsteps):
    """Run the sim while checking that the error is raised."""
    should_error = _using_gpu_with_cpu_device(
        sim.operations.integrator.forces[0].__class__, sim)
    if should_error:
        with pytest.raises(RuntimeError):
            sim.run(tsteps)
    else:
        sim.run(tsteps)
    return should_error


@pytest.mark.parametrize("force_cls", [MyForceCPU, MyForceGPU],
                         ids=lambda x: x.__name__)
def test_simulation(force_cls, simulation_factory, lattice_snapshot_factory):
    """Make sure custom force can plug into simulation without crashing."""
    snap = lattice_snapshot_factory()
    sim = simulation_factory(snap)
    _skip_if_cupy_not_imported_and_gpu_device(sim)
    custom_force = force_cls()
    nvt = md.methods.NPT(hoomd.filter.All(),
                         kT=1,
                         tau=1,
                         S=1,
                         tauS=1,
                         couple="none")
    integrator = md.Integrator(dt=0.005, forces=[custom_force], methods=[nvt])
    sim.operations.integrator = integrator
    exit_test = _try_running_sim(sim, 1)
    if exit_test:  # return if the sim can't be run
        return

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
        dot = np.dot(positions, b)
        cos_term = 1 / (2 * np.pi * self._p * self._w) * np.cos(self._p * dot)
        sin_term = 1 / (2 * np.pi * self._p * self._w) * np.sin(self._p * dot)
        energies = self._A * np.tanh(cos_term)
        forces = self._A * sin_term
        forces *= 1 - np.tanh(cos_term)**2
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
            arrays.force[:] = forces
            arrays.potential_energy[:] = potential


@pytest.mark.parametrize("force_cls", [MyPeriodicFieldCPU, MyPeriodicFieldGPU],
                         ids=lambda x: x.__name__)
def test_compare_to_periodic(force_cls, simulation_factory,
                             two_particle_snapshot_factory):
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
    sim2 = simulation_factory(snap2)
    _skip_if_cupy_not_imported_and_gpu_device(sim2)
    periodic2 = force_cls(A=1, i=0, p=1, w=1)
    nvt2 = md.methods.NVT(hoomd.filter.All(), kT=1, tau=1)
    integrator2 = md.Integrator(dt=0.005, forces=[periodic2], methods=[nvt2])
    sim2.operations.integrator = integrator2

    # run simulations next to each other
    sim.run(100)
    exit_test = _try_running_sim(sim2, 100)
    if exit_test:  # return if the sim can't be run
        return

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


class NestedForceCPU(md.force.Custom):

    def __init__(self):
        super().__init__()

    def set_forces(self, timestep):
        with self.cpu_local_force_arrays:
            with self.cpu_local_force_arrays:
                return


class NestedForceGPU(md.force.Custom):

    def __init__(self):
        super().__init__()

    def set_forces(self, timestep):
        with self.gpu_local_force_arrays:
            with self.gpu_local_force_arrays:
                return


@pytest.mark.parametrize("force_cls", [NestedForceCPU, NestedForceGPU],
                         ids=lambda x: x.__name__)
def test_nested_context_managers(force_cls, two_particle_snapshot_factory,
                                 simulation_factory):
    snap = two_particle_snapshot_factory()
    sim = simulation_factory(snap)
    custom_force = force_cls()
    nvt = md.methods.NPT(hoomd.filter.All(),
                         kT=1,
                         tau=1,
                         S=1,
                         tauS=1,
                         couple="none")
    integrator = md.Integrator(dt=0.005, forces=[custom_force], methods=[nvt])
    sim.operations.integrator = integrator
    with pytest.raises(RuntimeError):
        sim.run(1)


class GhostForceAccessCPU(md.force.Custom):

    def __init__(self, num_particles_global):
        super().__init__()
        self._num_particles_global = num_particles_global
        self._array_buffers = ['force', 'torque', 'potential_energy', 'virial']

    def _check_buffer_lengths(self, arrays):
        """Ensure the lengths of the accessed buffers are right."""
        for buffer_name in self._array_buffers:
            buffer = getattr(arrays, buffer_name)
            ghost_buffer = getattr(arrays, 'ghost_' + buffer_name)
            buffer_with_ghost = getattr(arrays, buffer_name + '_with_ghost')

            # make sure particle numbers add up wihtin the rank
            assert len(buffer) + len(ghost_buffer) == len(buffer_with_ghost)

            # make sure all particles are accounted for across ranks
            mpi_comm = MPI.COMM_WORLD
            N_global = mpi_comm.allreduce(len(buffer), op=MPI.SUM)
            assert N_global == self._num_particles_global

    def set_forces(self, timestep):
        with self.cpu_local_force_arrays as arrays:
            self._check_buffer_lengths(arrays)


class GhostForceAccessGPU(GhostForceAccessCPU):

    def __init__(self, num_ranks):
        super().__init__(num_ranks)

    def set_forces(self, timestep):
        with self.gpu_local_force_arrays as arrays:
            self._check_buffer_lengths(arrays)


@pytest.mark.parametrize("force_cls",
                         [GhostForceAccessCPU, GhostForceAccessGPU],
                         ids=lambda x: x.__name__)
def test_ghost_data_access(force_cls, two_particle_snapshot_factory,
                           simulation_factory):
    # skip this test if mpi4py not imported
    if not MPI4PY_IMPORTED:
        pytest.skip("This test needs mpi4py to run.")

    snap = two_particle_snapshot_factory()

    # split simulation so there is 1 particle in each rank
    sim = simulation_factory(snap, (2, 1, 1))
    if sim.device.communicator.rank == 0:
        global_N = snap.particles.N
    else:
        global_N = None
    mpi_comm = MPI.COMM_WORLD
    global_N = mpi_comm.bcast(global_N, root=0)
    custom_force = force_cls(global_N)

    # make LJ force so there is a ghost width on each rank
    nlist = md.nlist.Cell(buffer=0.2)
    lj_force = md.pair.LJ(nlist, default_r_cut=2.0)
    lj_force.params[('A', 'A')] = dict(sigma=1, epsilon=1)
    nvt = md.methods.NPT(hoomd.filter.All(),
                         kT=1,
                         tau=1,
                         S=1,
                         tauS=1,
                         couple="none")
    integrator = md.Integrator(dt=0.005,
                               forces=[custom_force, lj_force],
                               methods=[nvt])
    sim.operations.integrator = integrator
    _try_running_sim(sim, 2)
