# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest
import numpy as np
import numpy.testing as npt

# cupy works implicitly to set values in GPU force arrays
try:
    import cupy  # noqa F401
    CUPY_IMPORTED = True
except ImportError:
    # Necessary to test failure of using GPU buffers in CPU simulation.
    cupy = None
    CUPY_IMPORTED = False

# mpi4py is needed for the ghost data test
try:
    from mpi4py import MPI
    MPI4PY_IMPORTED = True
except ImportError:
    MPI4PY_IMPORTED = False

import hoomd
from hoomd import md


@pytest.fixture(scope='module')
def local_force_names(device):
    """Get local access properties based on the chosen devices."""
    names = ['cpu_local_force_arrays']
    if isinstance(device, hoomd.device.GPU):
        names.append('gpu_local_force_arrays')
    return names


@pytest.fixture(scope='module')
def force_simulation_factory(simulation_factory):
    """Create a basic simulation where there is only one force."""

    def make_sim(force_obj, snapshot=None, domain_decomposition=None):
        sim = simulation_factory(snapshot, domain_decomposition)
        thermostat = hoomd.md.methods.thermostats.MTTK(kT=1.0, tau=1.0)
        npt = md.methods.ConstantPressure(hoomd.filter.All(),
                                          S=1,
                                          tauS=1,
                                          couple="none",
                                          thermostat=thermostat)
        integrator = md.Integrator(dt=0.005, forces=[force_obj], methods=[npt])
        sim.operations.integrator = integrator
        return sim

    return make_sim


class MyEmptyForce(md.force.Custom):
    """Empty Force class; used by tests which access local buffers."""

    def __init__(self):
        super().__init__()

    def set_forces(self, timestep):
        pass


def _gpu_device_and_no_cupy(sim):
    gpu_device = isinstance(sim.device, hoomd.device.GPU)
    return gpu_device and not CUPY_IMPORTED


def _skip_if_gpu_device_and_no_cupy(sim):
    """Force classes cant set values if cupy isn't installed."""
    if _gpu_device_and_no_cupy(sim):
        pytest.skip("Cannot run this test on GPU without cupy")


class MyForce(md.force.Custom):

    def __init__(self, local_force_name):
        super().__init__(aniso=True)
        self._local_force_name = local_force_name

    def set_forces(self, timestep):
        if 'gpu' in self._local_force_name:
            array_mod = cupy
        else:
            array_mod = np
        with getattr(self, self._local_force_name) as arrays:
            arrays.force[:] = -5
            arrays.potential_energy[:] = 37
            arrays.torque[:] = 23
            if arrays.virial.shape[0] != 0:
                arrays.virial[:] = array_mod.arange(6)[None, :]


def test_simulation(local_force_names, force_simulation_factory,
                    lattice_snapshot_factory):
    """Make sure custom force can plug into simulation without crashing."""
    for local_force_name in local_force_names:
        snap = lattice_snapshot_factory()
        custom_force = MyForce(local_force_name)
        sim = force_simulation_factory(custom_force, snap)
        _skip_if_gpu_device_and_no_cupy(sim)
        sim.run(1)

        forces = custom_force.forces
        energies = custom_force.energies
        torques = custom_force.torques
        virials = custom_force.virials
        if sim.device.communicator.rank == 0:
            npt.assert_allclose(forces, -5)
            npt.assert_allclose(energies, 37)
            npt.assert_allclose(torques, 23)
            for i in range(6):
                npt.assert_allclose(virials[:, i], i)


class ForceAsFunctionOfTag(md.force.Custom):

    def __init__(self):
        super().__init__(aniso=True)

    def set_forces(self, timestep):
        with self.cpu_local_force_arrays as force_arrays:
            with self._state.cpu_local_snapshot as local_snapshot:
                tags = local_snapshot.particles.tag
                force_arrays.force[:] = np.stack((tags * 1, tags * 2, tags * 3),
                                                 axis=-1)
                energy = local_snapshot.particles.tag * -10
                force_arrays.potential_energy[:] = energy
                force_arrays.torque[:] = np.stack(
                    (tags * -3, tags * -2, tags * -1), axis=-1)
                if force_arrays.virial.shape[0] != 0:
                    force_arrays.virial[:] = np.stack(
                        (tags * 1, tags * -2, tags * -3, tags * 4, tags * -5,
                         tags * 6),
                        axis=-1)


@pytest.mark.cpu
def test_force_array_ordering(force_simulation_factory,
                              lattice_snapshot_factory):
    """Make sure values in force arrays are returned in correct order."""
    snap = lattice_snapshot_factory()
    custom_force = ForceAsFunctionOfTag()
    sim = force_simulation_factory(custom_force, snap)
    sim.run(1)

    forces = custom_force.forces
    energies = custom_force.energies
    torques = custom_force.torques
    virials = custom_force.virials
    indices = np.arange(sim.state.N_particles)
    if sim.device.communicator.rank == 0:
        npt.assert_array_equal(energies, np.arange(sim.state.N_particles) * -10)
        npt.assert_array_equal(
            forces, np.stack((indices * 1, indices * 2, indices * 3), axis=-1))
        npt.assert_array_equal(
            torques,
            np.stack((indices * -3, indices * -2, indices * -1), axis=-1))
        npt.assert_array_equal(
            virials,
            np.stack((indices * 1, indices * -2, indices * -3, indices * 4,
                      indices * -5, indices * 6),
                     axis=-1))


class MyPeriodicField(md.force.Custom):

    def __init__(self, local_force_name, A, i, p, w):
        super().__init__()
        self._local_force_name = local_force_name
        self._local_snap_name = local_force_name[:9] + "_snapshot"
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
        if 'gpu' in self._local_force_name:
            array_mod = cupy
        else:
            array_mod = np
        box = snapshot.global_box
        positions = self._numpy_array(snapshot.particles.position)

        # if no particles on this rank, return
        if positions.shape == (0,):
            return array_mod.array([]), array_mod.array([])

        a1, a2, a3 = box.to_matrix().T
        V = np.dot(a1, np.cross(a2, a3))
        b1 = 2 * np.pi / V * np.cross(a2, a3)
        b2 = 2 * np.pi / V * np.cross(a3, a1)
        b3 = 2 * np.pi / V * np.cross(a1, a2)
        b = {0: b1, 1: b2, 2: b3}.get(self._i)
        dot = array_mod.dot(array_mod.array(positions), array_mod.array(b))

        cos_term = 1 / (2 * array_mod.pi * self._p * self._w) * array_mod.cos(
            self._p * dot)
        sin_term = 1 / (2 * array_mod.pi * self._p * self._w) * array_mod.sin(
            self._p * dot)
        energies = self._A * array_mod.tanh(cos_term)
        forces = self._A * sin_term
        forces *= 1 - array_mod.tanh(cos_term)**2
        forces = array_mod.outer(forces, array_mod.array(b))
        return forces, energies

    def set_forces(self, timestep):
        with getattr(self._state, self._local_snap_name) as snap, \
                getattr(self, self._local_force_name) as arrays:
            forces, potential = self._evaluate_periodic(snap)
            arrays.force[:] = forces
            arrays.potential_energy[:] = potential


def test_compare_to_periodic(local_force_names, force_simulation_factory,
                             two_particle_snapshot_factory):
    """Test hoomd external periodic compared to a python version."""
    # sim with built-in force
    snap = two_particle_snapshot_factory()
    periodic = md.external.field.Periodic()
    periodic.params['A'] = dict(A=1, i=0, p=1, w=1)
    sim = force_simulation_factory(periodic, snap)
    integrator = sim.operations.integrator

    sim.run(100)
    snap_end = sim.state.get_snapshot()

    for local_force_name in local_force_names:
        # sim with custom but equivalent force
        snap2 = two_particle_snapshot_factory()
        periodic2 = MyPeriodicField(local_force_name, A=1, i=0, p=1, w=1)
        sim2 = force_simulation_factory(periodic2, snap2)
        _skip_if_gpu_device_and_no_cupy(sim2)
        integrator2 = sim2.operations.integrator

        sim2.run(100)
        snap_end2 = sim2.state.get_snapshot()

        # compare particle properties
        if sim.device.communicator.rank == 0:
            positions1 = snap_end.particles.position
            positions2 = snap_end2.particles.position
            npt.assert_allclose(positions1, positions2)

            velocities1 = snap_end.particles.velocity
            velocities2 = snap_end2.particles.velocity
            npt.assert_allclose(velocities1, velocities2)

        # compare force arrays
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

        virials1 = integrator.forces[0].virials
        virials2 = integrator2.forces[0].virials
        if sim.device.communicator.rank == 0:
            npt.assert_allclose(virials1, virials2)


def test_nested_context_managers(local_force_names,
                                 two_particle_snapshot_factory,
                                 force_simulation_factory):
    """Ensure we cannot nest local force context managers."""
    for local_force_name in local_force_names:
        snap = two_particle_snapshot_factory()
        custom_force = MyEmptyForce()
        sim = force_simulation_factory(custom_force, snap)
        sim.run(0)

        with pytest.raises(RuntimeError):
            with getattr(custom_force, local_force_name):
                with getattr(custom_force, local_force_name):
                    return


def test_ghost_data_access(local_force_names, two_particle_snapshot_factory,
                           force_simulation_factory):
    """Ensure size of ghost data arrays are correct."""
    # skip this test if mpi4py not imported
    if not MPI4PY_IMPORTED:
        pytest.skip("This test needs mpi4py to run.")

    for local_force_name in local_force_names:
        snap = two_particle_snapshot_factory()

        # split simulation so there is 1 particle in each rank
        custom_force = MyEmptyForce()
        sim = force_simulation_factory(custom_force, snap, (2, 1, 1))

        # make LJ force so there is a ghost width on each rank
        nlist = md.nlist.Cell(buffer=0.2)
        lj_force = md.pair.LJ(nlist, default_r_cut=2.0)
        lj_force.params[('A', 'A')] = dict(sigma=1, epsilon=1)
        sim.operations.integrator.forces.append(lj_force)
        sim.run(0)

        # each rank needs to know total number of particles
        if sim.device.communicator.rank == 0:
            N_global = snap.particles.N
        else:
            N_global = None
        mpi_comm = MPI.COMM_WORLD
        N_global = mpi_comm.bcast(N_global, root=0)

        # test buffer lengths
        array_buffers = ['force', 'torque', 'potential_energy', 'virial']
        with getattr(custom_force, local_force_name) as arrays:
            for buffer_name in array_buffers:
                buffer = getattr(arrays, buffer_name)
                ghost_buffer = getattr(arrays, 'ghost_' + buffer_name)
                buffer_with_ghost = getattr(arrays, buffer_name + '_with_ghost')

                # make sure particle numbers add up within the rank
                assert len(buffer) + len(ghost_buffer) == len(buffer_with_ghost)

                # make sure all particles are accounted for across ranks
                mpi_comm = MPI.COMM_WORLD
                N_global_computed = mpi_comm.allreduce(len(buffer), op=MPI.SUM)
                assert N_global_computed == N_global


def _assert_buffers_readonly(force_arrays):
    """Ensure proper errors are raised when trying to modify force buffers."""
    with pytest.raises(ValueError):
        force_arrays.force[:] = -100
    with pytest.raises(ValueError):
        force_arrays.virial[:] = 5
    with pytest.raises(ValueError):
        force_arrays.potential_energy[:] = -5
    with pytest.raises(ValueError):
        force_arrays.torque[:] = 2345


def test_data_buffers_readonly(local_force_names, two_particle_snapshot_factory,
                               simulation_factory):
    """Ensure local data buffers for non-custom force classes are read-only."""
    nlist = md.nlist.Cell(buffer=0.2)
    lj = md.pair.LJ(nlist, default_r_cut=2.0)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)

    snap = two_particle_snapshot_factory()
    sim = simulation_factory(snap)

    langevin = md.methods.Langevin(hoomd.filter.All(), kT=1)
    integrator = md.Integrator(dt=0.005, forces=[lj], methods=[langevin])
    sim.operations.integrator = integrator
    sim.run(2)

    for local_force_name in local_force_names:
        with getattr(lj, local_force_name) as arrays:
            if not _gpu_device_and_no_cupy(sim):
                _assert_buffers_readonly(arrays)


def _make_two_particle_snapshot(device, particle_types=['A'], d=1, L=20):
    """Make the snapshot.

    Args:
        device: hoomd device object.
        particle_types: List of particle type names
        dimensions: Number of dimensions (2 or 3)
        d: Distance apart to place particles
        L: Box length

    The two particles are placed at (-d/2, 0, 0) and (d/2,0,0). The box is L by
    L by L.
    """
    s = hoomd.Snapshot(device.communicator)

    if s.communicator.rank == 0:
        box = [L, L, L, 0, 0, 0]
        s.configuration.box = box
        s.particles.N = 2
        # shift particle positions slightly in z so MPI tests pass
        s.particles.position[:] = [[-d / 2, 0, .1], [d / 2, 0, .1]]
        s.particles.types = particle_types

    return s


def test_failure_with_cpu_device_and_gpu_buffer():
    """Assert we cannot access gpu buffers with a cpu_device."""
    device = hoomd.device.CPU()
    snap = _make_two_particle_snapshot(device)
    sim = hoomd.Simulation(device)
    sim.create_state_from_snapshot(snap)
    custom_force = MyForce('gpu_local_force_arrays')
    thermostat = hoomd.md.methods.thermostats.MTTK(kT=1.0, tau=1.0)
    npt = md.methods.ConstantPressure(hoomd.filter.All(),
                                      thermostat=thermostat,
                                      S=1,
                                      tauS=1,
                                      couple="none")
    integrator = md.Integrator(dt=0.005, forces=[custom_force], methods=[npt])
    sim.operations.integrator = integrator
    with pytest.raises(RuntimeError):
        sim.run(1)


def test_torques_update(local_force_names, two_particle_snapshot_factory,
                        force_simulation_factory):
    """Confirm torque'd particles' orientation changes over time."""
    initial_orientations = np.array([[1, 0, 0, 0], [1, 0, 0, 0]])
    for local_force_name in local_force_names:
        snap = two_particle_snapshot_factory()
        force = MyForce(local_force_name)
        if snap.communicator.rank == 0:
            snap.particles.moment_inertia[:] = [[1, 1, 1], [1, 1, 1]]
        sim = force_simulation_factory(force, snap)

        _skip_if_gpu_device_and_no_cupy(sim)
        sim.operations.integrator.integrate_rotational_dof = True

        if sim.device.communicator.rank == 0:
            npt.assert_allclose(snap.particles.orientation,
                                initial_orientations)
        sim.run(2)

        snap = sim.state.get_snapshot()
        if sim.device.communicator.rank == 0:
            assert np.count_nonzero(snap.particles.orientation
                                    - initial_orientations)


def test_force_zeroing(force_simulation_factory, two_particle_snapshot_factory):

    class TestForceZeroing(hoomd.md.force.Custom):

        def __init__(self):
            super().__init__(aniso=True)

        def set_forces(self, timestep):
            with self.cpu_local_force_arrays as force:
                force.force[:] += timestep
                force.torque[:] += timestep
                force.virial[:] += timestep

    snap = two_particle_snapshot_factory()
    test_force = TestForceZeroing()
    sim = force_simulation_factory(test_force, snap)
    if sim.device.communicator.rank == 0:
        snap.particles.moment_inertia[:] = [[1, 1, 1], [1, 1, 1]]
    sim.state.set_snapshot(snap)

    _skip_if_gpu_device_and_no_cupy(sim)
    sim.operations.integrator.integrate_rotational_dof = True
    sim.always_compute_pressure = True

    for _ in range(3):
        sim.run(1)
        forces = test_force.forces
        torques = test_force.torques
        virials = test_force.virials
        timestep = sim.timestep
        if sim.device.communicator.rank == 0:
            assert np.allclose(forces, timestep)
            assert np.allclose(torques, timestep)
            assert np.allclose(virials, timestep)
