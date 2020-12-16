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
    def _from_gsd_snapshot(cls, gsd_snapshot, communicator):
        snap = cls()
        # Set all particle attributes in snap from gsd_snapshot
        for key in vars(gsd_snapshot.particles):
            val = vars(gsd_snapshot.particles)[key]
            if val is not None:
                try:
                    setattr(snap.particles, key, val)
                except AttributeError:
                    try:
                        # Some attributes exist but aren't setable without [:]
                        x = getattr(snap.particles,key)
                        x[:] = val
                    except AttributeError:
                        # The attribute doesn't exist in hoomd.Snapshot
                        pass

        # Set all bond attributes
        for key in vars(gsd_snapshot.bonds):
            val = vars(gsd_snapshot.bonds)[key]
            if val is not None:
                print(key)
                try:
                    setattr(snap.bonds, key, val)
                except AttributeError:
                    try:
                        x = getattr(snap.bonds,key)
                        x[:] = val
                    except AttributeError:
                        pass

        # Set box attribute
        snap.configuration.box = gsd_snapshot.configuration.box
        return snap
