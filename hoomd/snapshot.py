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
        Constructs a `hoomd.Snapshot` from a `gsd.hoomd.Snapshot` object.

        Args:
            gsd_snap (`gsd.hoomd.Snapshot`):
                The gsd snapshot to convert to a `hoomd.Snapshot`.
            communicator (hoomd.communicator.Communicator):
                The MPI communicator to use for the snapshot. This prevents the
                snapshot from being stored on every rank.
        """
        gsd_snap.validate()
        snap = cls(communicator=communicator)

        def set_properties(
                snap_section, gsd_snap_section, properties, array_properties):
            for prop in properties:
                gsd_prop = getattr(gsd_snap_section, prop, None)
                if gsd_prop is not None:
                    setattr(snap_section, prop, gsd_prop)
            for prop in array_properties:
                gsd_prop = getattr(gsd_snap_section, prop, None)
                if gsd_prop is not None:
                    getattr(snap_section, prop)[:] = gsd_prop

        if communicator.rank == 0:

            set_properties(
                snap.particles,
                gsd_snap.particles,
                ('N', 'types'),
                ('angmom', 'body', 'charge', 'diameter', 'image', 'mass',
                 'moment_inertia', 'orientation', 'position', 'typeid',
                 'velocity')
            )

            for section in ('angles', 'bonds', 'dihedrals', 'improper', 'pair'):
                set_properties(
                    getattr(snap, section),
                    getattr(gsd_snap, section),
                    ('N', 'types'),
                    ('group', 'typeid')
                )

            set_properties(
                snap.constraints,
                gsd_snap.constraints,
                ('N',),
                ('group', 'value')
            )

            # Set box attribute
            if gsd_snap.configuration.box is not None:
                snap.configuration.box = gsd_snap.configuration.box
                if gsd_snap.configuration.dimensions == 2:
                    snap.configuration.box[2] = 0

        return snap
