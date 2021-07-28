"""Module implements the `State` class.

`State` stores and exposes a parent `hoomd.Simulation` object's data (e.g.
particle positions, system bonds).

"""
from collections import defaultdict

from . import _hoomd
from hoomd.box import Box
from hoomd.snapshot import Snapshot
from hoomd.data import LocalSnapshot, LocalSnapshotGPU
import hoomd
import warnings


def _create_domain_decomposition(device, box):
    """Create a default domain decomposition.

    This method is a quick hack to get basic MPI simulations working with
    the new API. We will need to consider designing an appropriate user-facing
    API to set the domain decomposition.
    """
    if not hoomd.version.mpi_enabled:
        return None

    # if we are only running on one processor, we use optimized code paths
    # for single-GPU execution
    if device.communicator.num_ranks == 1:
        return None

    # create a default domain decomposition
    result = _hoomd.DomainDecomposition(device._cpp_exec_conf, box.getL(), 0, 0,
                                        0, False)

    return result


class State:
    """The state of a `hoomd.Simulation` object.

    `State` stores a `hoomd.Simulation` object's particle, bond, angle, etc.
    data that describe the microstate of the system. Data access is facilitated
    through two complementary APIs: *local* snapshots that access data directly
    available on the local MPI rank and *global* snapshots that collect the
    entire state on rank 0. See `State.cpu_local_snapshot`,
    `State.gpu_local_snapshot`, `State.take_snapshot`, and
    `State.restore_snapshot` for information about these data access patterns.
    In addition, many commonly used quantities such as the number of particles
    and types in a simulation are available directly through `State` object
    properties.

    Note:
        This object should never be directly instantiated by users. There is no
        way to set a state created outside of a `hoomd.Simulation` object to a
        simulation. Use `hoomd.Simulation.create_state_from_gsd` and
        `hoomd.Simulation.create_state_from_snapshot` to instantiate a
        `State` object as part of a simulation.
    """

    def __init__(self, simulation, snapshot):
        self._simulation = simulation
        snapshot._broadcast_box()
        decomposition = _create_domain_decomposition(
            simulation.device, snapshot._cpp_obj._global_box)

        if decomposition is not None:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device._cpp_exec_conf,
                decomposition)
        else:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device._cpp_exec_conf)

        # Necessary for local snapshot API. This is used to ensure two local
        # snapshots are not contexted at once.
        self._in_context_manager = False

        # self._groups provides a cache of C++ group objects of the form:
        # {type(filter): {filter: C++ group}}
        # The first layer is to prevent user created filters with poorly
        # implemented __hash__ and __eq__ from causing cache errors.
        self._groups = defaultdict(dict)

    @property
    def snapshot(self):
        """Simulation snapshot.

        .. deprecated:: 3.0.0-beta.8
            Use `take_snapshot` and `restore_snapshot` instead.
        """
        warnings.warn("Deprecated, use state.take_snapshot()",
                      DeprecationWarning)
        return self.take_snapshot()

    @snapshot.setter
    def snapshot(self, snapshot):
        warnings.warn("Deprecated, use state.restore_snapshot()",
                      DeprecationWarning)
        self.restore_snapshot(snapshot)

    def take_snapshot(self):
        """Make a copy of the simulation current microstate.

        `State.take_snapshot` makes a copy of the simulation microstate and
        makes it available in a single object. `State.restore_snapshot` resets
        the internal microstate to that in the given snapshot. Use these methods
        to implement techniques like hybrid MD/MC or umbrella sampling where
        entire system configurations need to be reset to a previous one after a
        rejected move.

        Note:
            Data across all MPI ranks and from GPUs is gathered on the root MPI
            rank's memory. When accessing data in MPI simulations, use a ``if
            snapshot.communicator.rank == 0:`` conditional to access data arrays
            only on the root rank.

        Note:
            `State.take_snapshot` is an order :math:`O(N_{particles} + N_{bonds}
            + \\ldots)` operation.

        See Also:
            `restore_snapshot`

        Returns:
            hoomd.Snapshot: The current simulation microstate
        """
        cpp_snapshot = self._cpp_sys_def.takeSnapshot_double()
        return Snapshot._from_cpp_snapshot(cpp_snapshot,
                                           self._simulation.device.communicator)

    def restore_snapshot(self, snapshot):
        """Restore the microstate of the simulation from a snapshot.

        Args:
            snapshot (hoomd.Snapshot): Snapshot of the system from
              `take_snapshot`

        Warning:
            `restore_snapshot` can only make limited changes to the simulation
            state. While it can change the number of particles/bonds/etc... or
            their properties, it cannot change the number or names of the
            particle/bond/etc.. types.

        Note:
            `State.restore_snapshot` is an order :math:`O(N_{particles} +
            N_{bonds} + \\ldots)` operation and is very expensive when the
            simulation device is a GPU.

        See Also:
            `take_snapshot`
        """
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot set state to new snapshot inside local snapshot.")
        if self._simulation.device.communicator.rank == 0:
            if snapshot.particles.types != self.particle_types:
                raise RuntimeError(
                    "Particle types must remain the same")
            if snapshot.bonds.types != self.bond_types:
                raise RuntimeError("Bond types must remain the same")
            if snapshot.angles.types != self.angle_types:
                raise RuntimeError("Angle types must remain the same")
            if snapshot.dihedrals.types != self.dihedral_types:
                raise RuntimeError(
                    "Dihedral types must remain the same")
            if snapshot.impropers.types != self.improper_types:
                raise RuntimeError(
                    "Improper types must remain the same")
            if snapshot.pairs.types != self.special_pair_types:
                raise RuntimeError("Pair types must remain the same")

        self._cpp_sys_def.initializeFromSnapshot(snapshot._cpp_obj)

    @property
    def particle_types(self):
        """list[str]: List of all particle types in the simulation."""
        return self._cpp_sys_def.getParticleData().getTypes()

    @property
    def bond_types(self):
        """list[str]: List of all bond types in the simulation."""
        return self._cpp_sys_def.getBondData().getTypes()

    @property
    def angle_types(self):
        """list[str]: List of all angle types in the simulation."""
        return self._cpp_sys_def.getAngleData().getTypes()

    @property
    def dihedral_types(self):
        """list[str]: List of all dihedral types in the simulation."""
        return self._cpp_sys_def.getDihedralData().getTypes()

    @property
    def improper_types(self):
        """list[str]: List of all improper types in the simulation."""
        return self._cpp_sys_def.getImproperData().getTypes()

    @property
    def special_pair_types(self):
        """list[str]: List of all special pair types in the simulation."""
        return self._cpp_sys_def.getPairData().getTypes()

    @property
    def types(self):
        """dict[str, list[str]]: dictionary of all types in the state.

        Combines the data from `State.particle_types`, `State.bond_types`,
        `State.angle_types`, `State.dihedral_types`, `State.improper_types`, and
        `State.special_pair_types` into a dictionary with keys matching the
        property names.
        """
        return dict(particle_types=self.particle_types,
                    bond_types=self.bond_types,
                    angle_types=self.angle_types,
                    dihedral_types=self.dihedral_types,
                    improper_types=self.improper_types,
                    special_pair_types=self.special_pair_types)

    @property
    def N_particles(self):  # noqa: N802 - allow N in name
        """int: The number of particles in the simulation."""
        return self._cpp_sys_def.getParticleData().getNGlobal()

    @property
    def N_bonds(self):  # noqa: N802 - allow N in name
        """int: The number of bonds in the simulation."""
        return self._cpp_sys_def.getBondData().getNGlobal()

    @property
    def N_angles(self):  # noqa: N802 - allow N in name
        """int: The number of angles in the simulation."""
        return self._cpp_sys_def.getAngleData().getNGlobal()

    @property
    def N_impropers(self):  # noqa: N802 - allow N in name
        """int: The number of impropers in the simulation."""
        return self._cpp_sys_def.getImproperData().getNGlobal()

    @property
    def N_special_pairs(self):  # noqa: N802 - allow N in name
        """int: The number of special pairs in the simulation."""
        return self._cpp_sys_def.getPairData().getNGlobal()

    @property
    def N_dihedrals(self):  # noqa: N802 - allow N in name
        """int: The number of dihedrals in the simulation."""
        return self._cpp_sys_def.getDihedralData().getNGlobal()

    @property
    def box(self):
        """hoomd.Box: The current simulation box.

        Editing the box directly is not allowed. For example
        ``state.box.scale(1.1)`` would not scale the state's box. To set the
        state's box to a new box ``state.box = new_box`` must be used.
        """
        b = Box._from_cpp(self._cpp_sys_def.getParticleData().getGlobalBox())
        return Box.from_box(b)

    @box.setter
    def box(self, value):
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot set system box within local snapshot context manager.")
        try:
            value = Box.from_box(value)
        except Exception:
            raise ValueError('{} is not convertable to hoomd.Box using '
                             'hoomd.Box.from_box'.format(value))

        if value.dimensions != self._cpp_sys_def.getNDimensions():
            self._simulation.device._cpp_msg.warning(
                "Box changing dimensions from {} to {}."
                "".format(self._cpp_sys_def.getNDimensions(), value.dimensions))
            self._cpp_sys_def.setNDimensions(value.dimensions)
        self._cpp_sys_def.getParticleData().setGlobalBox(value._cpp_obj)

    def replicate(self):  # noqa: D102
        raise NotImplementedError

    def _get_group(self, filter_):
        cls = filter_.__class__
        if filter_ in self._groups[cls]:
            return self._groups[cls][filter_]
        else:
            if isinstance(filter_, hoomd.filter.CustomFilter):
                group = _hoomd.ParticleGroup(
                    self._cpp_sys_def,
                    _hoomd.ParticleFilterCustom(filter_, self))
            else:
                group = _hoomd.ParticleGroup(self._cpp_sys_def, filter_)
            self._groups[cls][filter_] = group
            return group

    def update_group_dof(self):
        """Update the number of degrees of freedom in each group.

        The groups of particles selected by filters each need to know the number
        of degrees of freedom given to that group by the simulation's
        Integrator. This method is called automatically when:

        * The Integrator is attached to the simulation

        Call it manually to force an update.
        """
        integrator = self._simulation.operations.integrator

        for groups in self._groups.values():
            for group in groups.values():
                if integrator is not None:
                    if not integrator._attached:
                        raise RuntimeError(
                            "Call update_group_dof after attaching")

                    integrator._cpp_obj.updateGroupDOF(group)
                else:
                    group.setTranslationalDOF(0)
                    group.setRotationalDOF(0)

    @property
    def cpu_local_snapshot(self):
        """hoomd.data.LocalSnapshot: Expose simulation data on the CPU.

        Provides access directly to the system state's particle, bond, angle,
        dihedral, improper, constaint, and pair data through a context manager.
        Data in `State.cpu_local_snapshot` is MPI rank local, and the
        `hoomd.data.LocalSnapshot` object is only usable within a context
        manager (i.e. ``with sim.state.cpu_local_snapshot as data:``). Attempts
        to assess data outside the context manager will result in errors. The
        local snapshot interface is similar to that of `hoomd.Snapshot`.

        The `hoomd.data.LocalSnapshot` data access is mediated through
        `hoomd.array.HOOMDArray` objects. This lets us ensure memory safety when
        directly accessing HOOMD-blue's data. The interface provides zero-copy
        access (zero-copy is guaranteed on CPU, access may be zero-copy if
        running on GPU).

        Changing the data in the buffers exposed by the local snapshot will
        change the data across the HOOMD-blue simulation. For a trivial example,
        this example would set all particle z-axis positions to 0.

        .. code-block:: python

            with sim.state.cpu_local_snapshot as data:
                data.particles.position[:, 2] = 0

        Note:
            The state's box and the number of particles, bonds, angles,
            dihedrals, impropers, constaints, and pairs cannot change within the
            context manager.

        Note:
            Getting a local snapshot object is order :math:`O(1)` and setting a
            single value is of order :math:`O(1)`.
        """
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot enter cpu_local_snapshot context manager inside "
                "another local_snapshot context manager.")
        return LocalSnapshot(self)

    @property
    def gpu_local_snapshot(self):
        """hoomd.data.LocalSnapshotGPU: Expose simulation data on the GPU.

        Provides access directly to the system state's particle, bond, angle,
        dihedral, improper, constaint, and pair data through a context manager.
        Data in `State.gpu_local_snapshot` is GPU local, and the
        `hoomd.data.LocalSnapshotGPU` object is only usable within a context
        manager (i.e. ``with sim.state.gpu_local_snapshot as data:``). Attempts
        to assess data outside the context manager will result in errors. The
        local snapshot interface is similar to that of `hoomd.Snapshot`.

        The `hoomd.data.LocalSnapshotGPU` data access is mediated through
        `hoomd.array.HOOMDGPUArray` objects. This helps us maintain memory
        safety when directly accessing HOOMD-blue's data. The interface provides
        zero-copy access on the GPU (assuming data was last accessed on the
        GPU).

        Changing the data in the buffers exposed by the local snapshot will
        change the data across the HOOMD-blue simulation. For a trivial example,
        this example would set all particle z-axis positions to 0.

        .. code-block:: python

            with sim.state.gpu_local_snapshot as data:
                data.particles.position[:, 2] = 0

        Warning:
            This property is only available when running on a GPU(s).

        Note:
            The state's box and the number of particles, bonds, angles,
            dihedrals, impropers, constaints, and pairs cannot change within the
            context manager.

        Note:
            Getting a local snapshot object is order :math:`O(1)` and setting a
            single value is of order :math:`O(1)`.
        """
        if not isinstance(self._simulation.device, hoomd.device.GPU):
            raise RuntimeError(
                "Cannot access gpu_snapshot with a non GPU device.")
        elif self._in_context_manager:
            raise RuntimeError(
                "Cannot enter gpu_local_snapshot context manager inside "
                "another local_snapshot context manager.")
        else:
            return LocalSnapshotGPU(self)

    def thermalize_particle_momenta(self, filter, kT):
        """Assign random values to particle momenta.

        Args:
            filter (hoomd.filter.ParticleFilter): Particles to modify
            kT (float): Thermal energy to set :math:`[\\mathrm{energy}]`

        `thermalize_particle_momenta` assigns the selected particle's velocities
        and angular momentum to random values drawn from a Gaussian distribution
        consistent with the given thermal energy *kT*.

        .. rubric:: Velocity

        `thermalize_particle_momenta` assigns random velocities to the *x* and
        *y* components of each particle's velocity. When the simulation box is
        3D, it also assigns a random velocity to the *z* component. When the
        simulation box is 2D, it sets the *z* component to 0. Finally,
        sets the center of mass velocity of the selected particles to 0.

        .. rubric:: Angular momentum

        `thermalize_particle_momenta` assigns random angular momenta to each
        rotational degree of freedom that has a non-zero moment of intertia.
        Each particle can have 0, 1, 2, or 3 rotational degrees of freedom
        as determine by its moment of inertia.

        .. seealso::
            `md.methods.NVT.thermalize_thermostat_dof`

            `md.methods.NPT.thermalize_thermostat_and_barostat_dof`
        """
        self._simulation._warn_if_seed_unset()
        group = self._get_group(filter)
        group.thermalizeParticleMomenta(kT, self._simulation.timestep)
