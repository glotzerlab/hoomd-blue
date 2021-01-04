import numpy as np
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
    def from_gsd_snapshot(cls, gsd_snap, communicator):
        """
        Constructs a hoomd.Snapshot from a gsd.hoomd.Snapshot
        """
        gsd_snap.validate()
        snap = cls(communicator=communicator)
        if communicator.rank == 0:

            # Set all particle attributes in snap from gsd_snap
            for property_ in ('N', 'types'):
                gsd_prop = getattr(gsd_snap.particles, property_)
                if gsd_prop is not None:
                    setattr(snap.particles, property_, gsd_prop)
            for property_ in (
                    'angmom', 'body', 'charge', 'diameter', 'image', 'mass',
                    'moment_inertia', 'orientation', 'position', 'typeid',
                    'velocity'
                    ):
                gsd_prop = getattr(gsd_snap.particles, property_)
                if gsd_prop is not None:
                    hoomd_prop = getattr(snap.particles, property_)
                    hoomd_prop[:] = gsd_prop

            # Set all bond attributes
            for property_ in ('N', 'types'):
                gsd_prop = getattr(gsd_snap.bonds, property_)
                if gsd_prop is not None:
                    setattr(snap.bonds, property_, gsd_prop)
            for property_ in ('group', 'typeid'):
                gsd_prop = getattr(gsd_snap.bonds, property_)
                if gsd_prop is not None:
                    hoomd_prop = getattr(snap.bonds, property_)
                    hoomd_prop[:] = gsd_prop

            # Set all angle attributes
            for property_ in ('N', 'types'):
                gsd_prop = getattr(gsd_snap.angles, property_)
                if gsd_prop is not None:
                    setattr(snap.angles, property_, gsd_prop)
            for property_ in ('group', 'typeid'):
                gsd_prop = getattr(gsd_snap.angles, property_)
                if gsd_prop is not None:
                    hoomd_prop = getattr(snap.angles, property_)
                    hoomd_prop[:] = gsd_prop

            # Set all dihedral attributes
            for property_ in ('N', 'types'):
                gsd_prop = getattr(gsd_snap.dihedrals, property_)
                if gsd_prop is not None:
                    setattr(snap.dihedrals, property_, gsd_prop)
            for property_ in ('group', 'typeid'):
                gsd_prop = getattr(gsd_snap.dihedrals, property_)
                if gsd_prop is not None:
                    hoomd_prop = getattr(snap.dihedrals, property_)
                    hoomd_prop[:] = gsd_prop

            # Set all improper attributes
            for property_ in ('N', 'types'):
                gsd_prop = getattr(gsd_snap.impropers, property_)
                if gsd_prop is not None:
                    setattr(snap.impropers, property_, gsd_prop)
            for property_ in ('group', 'typeid'):
                gsd_prop = getattr(gsd_snap.impropers, property_)
                if gsd_prop is not None:
                    hoomd_prop = getattr(snap.impropers, property_)
                    hoomd_prop[:] = gsd_prop

            # Set all pair attributes
            for property_ in ('N', 'types'):
                gsd_prop = getattr(gsd_snap.pairs, property_)
                if gsd_prop is not None:
                    setattr(snap.pairs, property_, gsd_prop)
            for property_ in ('group', 'typeid'):
                gsd_prop = getattr(gsd_snap.pairs, property_)
                if gsd_prop is not None:
                    hoomd_prop = getattr(snap.pairs, property_)
                    hoomd_prop[:] = gsd_prop

            # Set all constraint attributes
            gsd_prop = getattr(gsd_snap.constraints, 'N')
            if gsd_prop is not None:
                setattr(snap.constraints, 'N', gsd_prop)
            for property_ in ('group', 'value'):
                gsd_prop = getattr(gsd_snap.constraints, property_)
                if gsd_prop is not None:
                    hoomd_prop = getattr(snap.constraints, property_)
                    hoomd_prop[:] = gsd_prop

            # Set box attribute
            if gsd_snap.configuration.box is not None:
                snap.configuration.box = gsd_snap.configuration.box
                if gsd_snap.configuration.dimensions == 2:
                    box = np.array(snap.configuration.box)
                    box[3] = 0
                    snap.configuration.box = box
        return snap
