from . import _hoomd
from .data import boxdim


class State:
    R"""
    Parameters:
        device
        snapshot

    Attributes:
    """

    def __init__(self, device, snapshot):
        self._device = device
        snapshot._broadcast_box(device.cpp_exec_conf)
        # my_domain_decomposition = _create_domain_decomposition(snapshot._global_box);

        if False:  # my_domain_decomposition is not None:
            self._cpp_system_definition = _hoomd.SystemDefinition(
                snapshot, device.cpp_exec_conf, my_domain_decomposition)
        else:
            self._cpp_system_definition = _hoomd.SystemDefinition(
                snapshot, device.cpp_exec_conf)

    @property
    def snapshot(self):
        return self._cpp_system_definition.takeSnapshot_double(*([True] * 8))

    @property
    def light_snapshot(self):
        return self._cpp_system_definition.takeSnapshot_float(*([True] * 8))

    @snapshot.setter
    def restore_snapshot(self, snapshot):
        R""" Re-initializes the system from a snapshot.

        Args:
            snapshot:. The snapshot to initialize the system from.

        Snapshots temporarily store system data. Snapshots contain the complete simulation state in a
        single object. They can be used to restart a simulation.

        Example use cases in which a simulation may be restarted from a snapshot include python-script-level
        Monte-Carlo schemes, where the system state is stored after a move has been accepted (according to
        some criterion), and where the system is re-initialized from that same state in the case
        when a move is not accepted.

        Example::

            system = init.read_xml("some_file.xml")

            ... run a simulation ...

            snapshot = system.take_snapshot(all=True)
            ...
            system.restore_snapshot(snapshot)

        Warning:
                restore_snapshot() may invalidate force coefficients, neighborlist r_cut values, and other per type
                quantities if called within a callback during a run(). You can restore a snapshot during a run only
                if the snapshot is of a previous state of the currently running system. Otherwise, you need to use
                restore_snapshot() between run() commands to ensure that all per type coefficients are updated properly.

        """

        if self._device.comm.rank == 0:
            if snapshot.has_particle_data and \
                    len(snapshot.particles.types) != self.ntypes:
                raise RuntimeError(
                    "Number of particle types must remain the same")
            if snapshot.has_bond_data and \
                    len(snapshot.bonds.types) != self.ntype_bonds:
                raise RuntimeError("Number of bond types must remain the same")
            if snapshot.has_angle_data and \
                    len(snapshot.angles.types) != self.ntype_angles:
                raise RuntimeError(
                    "Number of angle types must remain the same")
            if snapshot.has_dihedral_data and \
                    len(snapshot.dihedrals.types) != self.ntype_dihedral:
                raise RuntimeError(
                    "Number of dihedral types must remain the same")
            if snapshot.has_improper_data and \
                    len(snapshot.impropers.types) != self.ntype_impropers:
                raise RuntimeError(
                    "Number of dihedral types must remain the same")
            if snapshot.has_pair_data and \
                    len(snapshot.pairs.types) != self.ntype_pairs:
                raise RuntimeError("Number of pair types must remain the same")

        self._cpp_system_definition.initializeFromSnapshot(snapshot)

    @property
    def ntypes(self):
        return self._cpp_system_definition.getParticleData().getNTypes()

    @property
    def ntype_bonds(self):
        return self._cpp_system_definition.getBondData().getNTypes()

    @property
    def ntype_angles(self):
        return self._cpp_system_definition.getAngleData().getNTypes()

    @property
    def ntype_dihedral(self):
        return self._cpp_system_definition.getDihedralData().getNTypes()

    @property
    def ntype_impropers(self):
        return self._cpp_system_definition.getImproperData().getNTypes()

    @property
    def ntype_pairs(self):
        return self._cpp_system_definition.getPairData().getNTypes()

    @property
    def box(self):
        b = self._cpp_system_definition.getParticleData().getGlobalBox()
        L = b.getL()
        return boxdim(Lx=L.x, Ly=L.y, Lz=L.z,
                      xy=b.getTiltFactorXY(),
                      xz=b.getTiltFactorXZ(),
                      yz=b.getTiltFactorYZ(),
                      dimensions=self._cpp_system_definition.getNDimensions())

    # Set the system box
    # \param value The new boundaries (a data.boxdim object)
    @box.setter
    def box(self, value):
        if not isinstance(value, boxdim):
            raise TypeError('box must be a data.boxdim object')
        self._cpp_system_definition.getParticleData().setGlobalBox(value._getBoxDim())

    def replicate(self):
        raise NotImplementedError

    def scale_system(self):
        raise NotImplementedError
