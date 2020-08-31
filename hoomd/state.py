from copy import copy
from collections import defaultdict

from . import _hoomd
from hoomd.box import Box
from hoomd.snapshot import Snapshot
from hoomd.data import LocalSnapshot, LocalSnapshotGPU


def _create_domain_decomposition(device, box):
    """ Create a default domain decomposition.

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
    result = _hoomd.DomainDecomposition(device.cpp_exec_conf,
                                        box.getL(),
                                        0,
                                        0,
                                        0,
                                        False)

    return result


class State:
    R""" Simulation state.

    Args:
        snapshot
    """

    def __init__(self, simulation, snapshot):
        self._simulation = simulation
        self._in_context_manager = False
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
        self._groups = defaultdict(dict)

    @property
    def snapshot(self):
        cpp_snapshot = self._cpp_sys_def.takeSnapshot_double()
        return Snapshot._from_cpp_snapshot(cpp_snapshot,
                                           self._simulation.device.communicator)

    @snapshot.setter
    def snapshot(self, snapshot):
        R""" Re-initializes the system from a snapshot.

        Args:
            snapshot:. The snapshot to initialize the system from.

        Snapshots temporarily store system data. Snapshots contain the complete
        simulation state in a single object. They can be used to restart a
        simulation.

        Example use cases in which a simulation may be restarted from a snapshot
        include python-script-level Monte-Carlo schemes, where the system state
        is stored after a move has been accepted (according to some criterion),
        and where the system is re-initialized from that same state in the case
        when a move is not accepted.

        Example::

            system = init.read_xml("some_file.xml")

            ... run a simulation ...

            snapshot = system.take_snapshot(all=True)
            ...
            system.restore_snapshot(snapshot)

        Warning:
                restore_snapshot() may invalidate force coefficients,
                neighborlist r_cut values, and other per type quantities if
                called within a callback during a run(). You can restore a
                snapshot during a run only if the snapshot is of a previous
                state of the currently running system. Otherwise, you need to
                use restore_snapshot() between run() commands to ensure that all
                per type coefficients are updated properly.

        """

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
    def types(self):
        return dict(particle_types=self.particle_types,
                    bond_types=self.bond_types,
                    angle_types=self.angles_types,
                    dihedral_types=self.dihedral_types,
                    improper_types=self.improper_types,
                    special_pair_types=self.special_pair_types
                    )

    @property
    def particle_types(self):
        return self._cpp_sys_def.getParticleData().getTypes()

    @property
    def bond_types(self):
        return self._cpp_sys_def.getBondData().getTypes()

    @property
    def angle_types(self):
        return self._cpp_sys_def.getAngleData().getTypes()

    @property
    def dihedral_types(self):
        return self._cpp_sys_def.getDihedralData().getTypes()

    @property
    def improper_types(self):
        return self._cpp_sys_def.getImproperData().getTypes()

    @property
    def special_pair_types(self):
        return self._cpp_sys_def.getPairData().getTypes()

    @property
    def box(self):
        """The state's box (a :py:class:`hoomd.Box` object).

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

    def replicate(self):
        raise NotImplementedError

    def scale_system(self):
        raise NotImplementedError

    def get_group(self, filter_):
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
                    if not integrator.is_attached:
                        raise RuntimeError(
                            "Call update_group_dof after attaching")

                    integrator._cpp_obj.updateGroupDOF(group)
                else:
                    group.setTranslationalDOF(0)
                    group.setRotationalDOF(0)

    @property
    def cpu_local_snapshot(self):
        """hoomd.data.LocalSnapshot: Directly expose HOOMD data buffers
        on the CPU.

        Provides access directly to the system state's particle, bond, angle,
        dihedral, improper, constaint, and pair data through a context manager.
        The `hoomd.data.LocalSnapshot` object is only usable within a
        context manager (i.e. ``with sim.state.cpu_local_snapshot as data:``)
        The interface is similar to that of the `hoomd.Snapshot`. Data is local
        to a given MPI rank. The returned arrays are `hoomd.array.HOOMDArray`
        objects. Through this interface zero-copy access is available (access is
        guarenteed to be zero-copy when running on CPU, may be zero copy if
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
        """hoomd.data.LocalSnapshotGPU: Directly expose HOOMD data
        buffers on the GPU.

        Provides access directly to the system state's particle, bond, angle,
        dihedral, improper, constaint, and pair data through a context manager.
        The `hoomd.data.LocalSnapshotGPU` object is only usable
        within a context manager (i.e. ``with sim.state.gpu_local_snapshot as
        data:``) The interface is similar to that of the `hoomd.Snapshot`. Data
        is local to a given MPI rank. The returned arrays are
        `hoomd.array.HOOMDGPUArray` objects. Through this interface potential
        zero-copy access is available (access cannot be guarenteed to be
        zero-copy, but will be if the most recent copy of the data is on the
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
        if self._simulation.device.mode != 'gpu':
            raise RuntimeError(
                "Cannot access gpu_snapshot with a non GPU device.")
        elif self._in_context_manager:
            raise RuntimeError(
                "Cannot enter gpu_local_snapshot context manager inside "
                "another local_snapshot context manager.")
        else:
            return LocalSnapshotGPU(self)
