from . import _hoomd
from hoomd.groups import Groups
from .data import boxdim
from hoomd.snapshot import Snapshot


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
    if device.comm.num_ranks == 1:
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
    R"""
    Parameters:
        simulation
        snapshot

    Attributes:
    """

    def __init__(self, simulation, snapshot):
        self._simulation = simulation
        snapshot._broadcast_box()
        domain_decomp = _create_domain_decomposition(
            simulation.device,
            snapshot._cpp_obj._global_box)

        if domain_decomp is not None:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device.cpp_exec_conf,
                domain_decomp)
        else:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device.cpp_exec_conf)
        self._groups = Groups()

    @property
    def snapshot(self):
        cpp_snapshot = self._cpp_sys_def.takeSnapshot_double()
        return Snapshot._from_cpp_snapshot(cpp_snapshot,
                                           self._simulation.device.comm)

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

        if self._simulation.device.comm.rank == 0:
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
        b = self._cpp_sys_def.getParticleData().getGlobalBox()
        L = b.getL()
        return boxdim(Lx=L.x, Ly=L.y, Lz=L.z,
                      xy=b.getTiltFactorXY(),
                      xz=b.getTiltFactorXZ(),
                      yz=b.getTiltFactorYZ(),
                      dimensions=self._cpp_sys_def.getNDimensions())

    # Set the system box
    # \param value The new boundaries (a data.boxdim object)
    @box.setter
    def box(self, value):
        if not isinstance(value, boxdim):
            raise TypeError('box must be a data.boxdim object')
        self._cpp_sys_def.getParticleData().setGlobalBox(value._getBoxDim())

    def replicate(self):
        raise NotImplementedError

    def scale_system(self):
        raise NotImplementedError

    def add_group(self, filter_):
        group = self._groups[filter_] = _hoomd.ParticleGroup(
            self._cpp_sys_def, filter_)
        return group
