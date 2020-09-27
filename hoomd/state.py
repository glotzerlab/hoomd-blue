"""Module implments the `State` class.

`State` stores and exposes a parent `hoomd.Simulation` object's data (e.g.
particle positions, system bonds).

"""
from collections import defaultdict

from . import _hoomd
from hoomd.box import Box
from hoomd.snapshot import Snapshot
from hoomd.data import LocalSnapshot, LocalSnapshotGPU
import hoomd


def _create_domain_decomposition(device, box):
    """Create a default domain decomposition.

    This method is a quick hack to get basic MPI simulations working with
    the new API. We will need to consider designing an appropriate user-facing
    API to set the domain decomposition.
    """
    if not _hoomd.is_MPI_available():
        return None

    # if we are only running on one processor, we use optimized code paths
    # for single-GPU execution
    if device.communicator.num_ranks == 1:
        return None

    # create a default domain decomposition
    result = _hoomd.DomainDecomposition(device._cpp_exec_conf,
                                        box.getL(),
                                        0,
                                        0,
                                        0,
                                        False)

    return result


class State:
    """The state of a `hoomd.Simulation` object.

    Provides access (read/write) to a `hoomd.Simulation` object's particle,
    bond, angle, etc. data. Data access is facilitated through two complementary
    APIs: *global* and *local* snapshots (note that global does not refer to
    variable scope here). See `State.snapshot`, `State.cpu_local_snapshot`, and
    `State.gpu_local_snapshot` for information about these data access patterns.
    In addition, many commonly used smaller quantites such as the number of
    particles in a simulation are available directly through `State` object
    properties. Accessing these quantities through a `State` object directly is
    a bit faster than accessing them through a local snapshot and much faster
    than accessing them through `State.snapshot`.

    Note:
        This object should never be directly instantiated by users. There is no
        way to set a state created outside of a `hoomd.Simulation` object to a
        simulation. See `hoomd.Simulation.create_state_from_gsd` and
        `hoomd.Simulation.create_state_from_snapshot` for information about
        instantiating `State` objects.
    """

    def __init__(self, simulation, snapshot):
        self._simulation = simulation
        snapshot._broadcast_box()
        domain_decomp = _create_domain_decomposition(
            simulation.device,
            snapshot._cpp_obj._global_box)

        if domain_decomp is not None:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device._cpp_exec_conf,
                domain_decomp)
        else:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device._cpp_exec_conf)

        # Necessary for local snapshot API. This is used to ensure two local
        # snapshots are not contexted at once.
        self._in_context_manager = False

        # provides a cache of C++ group objects of the form {type(filter):
        # {filter: C++ group}}. The first layer is to prevent user created
        # filters with poorly implemented __hash__ and __eq__ from causing cache
        # errors.
        self._groups = defaultdict(dict)

    @property
    def snapshot(self):
        """hoomd.Snapshot: All data of a simulation's current microstate.

        `State.snapshot` should be used when all of a simulation's state
        information is desired in a single object. When accessed, data across
        all MPI ranks and from GPUs is gathered on the root MPI rank's memory.
        When accessing data in MPI simulations then it is recommended to use a
        ``if snapshot.exists:`` conditional to prevent attempting to access data
        on a non-root rank.

        This property can be set to the state using an entirely new
        `hoomd.Snapshot` object.  Example use cases in which a simulation's
        state may be reset from a snapshot include python-script-level
        Monte-Carlo schemes, where the current snapshot is passed to the
        Monte-Carlo simulation before being passed back after running some
        Monte-Carlo steps.

        Warning:
            Using `State.snapshot` multiple times will gather data across MPI
            ranks and GPUs every time. If the snapshot is needed for more than
            one use case store it in a variable.

        Note:
            For performance critical use cases that don't benefit from having
            the full aggregated data, the local snapshot API
            (`State.cpu_local_snapshot` and `State.gpu_local_snapshot`) is
            recommended. This is most often the case when frequently accessing
            and modifying the simulation data in Python in a
            `hoomd.custom.Action`. In such scenarios, the local snapshot would
            likely perform much better with the local snapshot API. Only a few
            accesses or single extensive modifications would still be faster
            using the local snapshot, but is likely not to matter given the time
            of a typical simulation.
        """
        cpp_snapshot = self._cpp_sys_def.takeSnapshot_double()
        return Snapshot._from_cpp_snapshot(cpp_snapshot,
                                           self._simulation.device.communicator)

    @snapshot.setter
    def snapshot(self, snapshot):
        if self._simulation.device.communicator.rank == 0:
            if len(snapshot.particles.types) != len(self.particle_types):
                raise RuntimeError(
                    "Number of particle types must remain the same")
            if len(snapshot.bonds.types) != len(self.bond_types):
                raise RuntimeError("Number of bond types must remain the same")
            if len(snapshot.angles.types) != len(self.angle_types):
                raise RuntimeError(
                    "Number of angle types must remain the same")
            if len(snapshot.dihedrals.types) != len(self.dihedral_types):
                raise RuntimeError(
                    "Number of dihedral types must remain the same")
            if len(snapshot.impropers.types) != len(self.improper_types):
                raise RuntimeError(
                    "Number of dihedral types must remain the same")
            if len(snapshot.pairs.types) != len(self.special_pair_types):
                raise RuntimeError("Number of pair types must remain the same")

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
                    angle_types=self.angles_types,
                    dihedral_types=self.dihedral_types,
                    improper_types=self.improper_types,
                    special_pair_types=self.special_pair_types
                    )

    @property
    def N_particles(self):
        """int: The number of particles in the simulation."""
        return self._cpp_sys_def.getParticleData().getNGlobal()

    @property
    def N_bonds(self):
        """int: The number of bonds in the simulation."""
        return self._cpp_sys_def.getBondData().getNGlobal()

    @property
    def N_angles(self):
        """int: The number of angles in the simulation."""
        return self._cpp_sys_def.getAngleData().getNGlobal()

    @property
    def N_impropers(self):
        """int: The number of impropers in the simulation."""
        return self._cpp_sys_def.getImproperData().getNGlobal()

    @property
    def N_special_pairs(self):
        """int: The number of special pairs in the simulation."""
        return self._cpp_sys_def.getPairData().getNGlobal()

    @property
    def N_dihedrals(self):
        """int: The number of dihedrals in the simulation."""
        return self._cpp_sys_def.getDihedralData().getNGlobal()

    @property
    def box(self):
        """hoomd.Box: The current simulation box.

        Ediing the box directly is not allowed.  For example
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
                "".format(self._cpp_sys_def.getNDimensions(),
                          value.dimensions))
            self._cpp_sys_def.setNDimensions(value.dimensions)
        self._cpp_sys_def.getParticleData().setGlobalBox(value._cpp_obj)

    def replicate(self):  # noqa: D102
        raise NotImplementedError

    def _get_group(self, filter_):
        cls = filter_.__class__
        if filter_ in self._groups[cls]:
            return self._groups[cls][filter_]
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
        manager (i.e. ``with sim.state.cpu_local_snapshot as data:``).  Attempts
        to assess data outside the context manager will result in errors.  The
        local snapshot interface is similar to that of `hoomd.Snapshot`.

        The `hoomd.data.LocalSnapshot` data access is mediated through
        `hoomd.array.HOOMDArray` objects. This lets us ensure memory safety when
        directly accessing HOOMD-blue's data. The interface provides zero-copy
        access (zero-copy is guarenteed on CPU, access may be zero copy if
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
        manager (i.e. ``with sim.state.gpu_local_snapshot as data:``).  Attempts
        to assess data outside the context manager will result in errors.  The
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

        Note:
            The state's box and the number of particles, bonds, angles,
            dihedrals, impropers, constaints, and pairs cannot change within the
            context manager.

        Note:
            This property is only available when running on a GPU(s).
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
