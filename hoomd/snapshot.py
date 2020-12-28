import hoomd
from hoomd import _hoomd


class _ConfigurationData:
    def __init__(self, cpp_obj):
        self._cpp_obj = cpp_obj

    @property
    def dimensions(self):
        return self._cpp_obj._dimensions

    @property
    def box(self):
        b = self._cpp_obj._global_box
        L = b.getL()
        return (L.x, L.y, L.z,
                b.getTiltFactorXY(),
                b.getTiltFactorXZ(),
                b.getTiltFactorYZ())

    @box.setter
    def box(self, box):
        try:
            new_box = hoomd.Box.from_box(box)
        except Exception:
            raise ValueError(
                f"{box} is not convertible to a hoomd.Box object using "
                "hoomd.Box.from_box.")
        self._cpp_obj._dimensions = new_box.dimensions
        self._cpp_obj._global_box = new_box._cpp_obj


class Snapshot:
    def __init__(self, communicator=None):
        if communicator is None:
            self._comm = hoomd.communicator.Communicator()
        else:
            self._comm = communicator

        self._cpp_obj = _hoomd.SnapshotSystemData_double()

    @property
    def exists(self):
        return self._comm.rank == 0

    @property
    def configuration(self):
        return _ConfigurationData(self._cpp_obj)

    @property
    def particles(self):
        if self.exists:
            return self._cpp_obj.particles
        else:
            return None

    @property
    def bonds(self):
        if self.exists:
            return self._cpp_obj.bonds
        else:
            return None

    @property
    def angles(self):
        if self.exists:
            return self._cpp_obj.angles
        else:
            return None

    @property
    def dihedrals(self):
        if self.exists:
            return self._cpp_obj.dihedrals
        else:
            return None

    @property
    def impropers(self):
        if self.exists:
            return self._cpp_obj.impropers
        else:
            return None

    @property
    def pairs(self):
        if self.exists:
            return self._cpp_obj.pairs
        else:
            return None

    @property
    def constraints(self):
        if self.exists:
            return self._cpp_obj.constraints
        else:
            return None

    @classmethod
    def _from_cpp_snapshot(cls, snapshot, communicator):
        sp = cls()
        sp._comm = communicator
        sp._cpp_obj = snapshot
        return sp

    def replicate(self, nx, ny, nz):
        self._cpp_obj.replicate(nx, ny, nz)

    def _broadcast_box(self):
        self._cpp_obj._broadcast_box(self._comm.cpp_mpi_conf)

    @classmethod
    def _from_gsd_snapshot(cls, gsd_snap):
        gsd_snap.validate()
        if self._device.communicator.rank == 0:
            snap = cls(communicator=self._device.communicator)

            # Set all particle attributes in snap from gsd_snap
            snap.particles.N = gsd_snap.particles.N
            snap.particles.types = gsd_snap.particles.types
            snap.particles.angmom[:] = gsd_snap.particles.angmom
            snap.particles.body[:] = gsd_snap.particles.body
            snap.particles.charge[:] = gsd_snap.particles.charge
            snap.particles.diameter[:] = gsd_snap.particles.diameter
            snap.particles.image[:] = gsd_snap.particles.image
            snap.particles.mass[:] = gsd_snap.particles.mass
            snap.particles.moment_inertia[:] = gsd_snap.particles.moment_inertia
            snap.particles.orientation[:] = gsd_snap.particles.orientation
            snap.particles.position[:] = gsd_snap.particles.position
            snap.particles.typeid[:] = gsd_snap.particles.typeid
            snap.particles.velocity[:] = gsd_snap.particles.velocity

            # Set all bond attributes
            snap.bonds.N = gsd_snap.bonds.N
            snap.bonds.types = gsd_snap.bonds.types
            snap.bonds.group[:] = gsd_snap.bonds.group
            snap.bonds.typeid[:] = gsd_snap.bonds.typeid

            # Set all angle attributes
            snap.angles.N = gsd_snap.angles.N
            snap.angles.types = gsd_snap.angles.types
            snap.angles.group[:] = gsd_snap.angles.group
            snap.angles.typeid[:] = gsd_snap.angles.typeid

            # Set all dihedral attributes
            snap.dihedrals.N = gsd_snap.dihedrals.N
            snap.dihedrals.types = gsd_snap.dihedrals.types
            snap.dihedrals.group[:] = gsd_snap.dihedrals.group
            snap.dihedrals.typeid[:] = gsd_snap.dihedrals.typeid

            # Set all improper attributes
            snap.impropers.N = gsd_snap.impropers.N
            snap.impropers.types = gsd_snap.impropers.types
            snap.impropers.group[:] = gsd_snap.impropers.group
            snap.impropers.typeid[:] = gsd_snap.impropers.typeid

            # Set all pair attributes
            snap.pairs.N = gsd_snap.pairs.N
            snap.pairs.types = gsd_snap.pairs.types
            snap.pairs.group[:] = gsd_snap.pairs.group
            snap.pairs.typeid[:] = gsd_snap.pairs.typeid

            # Set all constraint attributes
            snap.constraints.N = gsd_snap.constraints.N
            snap.constraints.group[:] = gsd_snap.constraints.group
            snap.constraints.value[:] = gsd_snap.constraints.value

            # Set box attribute
            snap.configuration.box = gsd_snap.configuration.box
            if gsd_snap.configuration.dimensions == 2:
                snap.configuration.box[3] = 0
        return snap
