# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement Snapshot.

.. invisible-code-block: python

    # prepare snapshot and gsd file for later examples
    simulation = hoomd.util.make_example_simulation()

    snapshot = simulation.state.get_snapshot()
"""

import hoomd
from hoomd import _hoomd
import warnings


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
        return (L.x, L.y, L.z, b.getTiltFactorXY(), b.getTiltFactorXZ(),
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
    """Self-contained copy of the simulation `State`.

    Args:
        communicator (Communicator): MPI communicator to be used when accessing
            the snapshot.

    See `State` and `gsd.hoomd.Frame` for detailed documentation on the
    components of `Snapshot`.

    Note:
        `Snapshot` is duck-type compatible with `gsd.hoomd.Frame` except
        that arrays in `Snapshot` are not assignable. You can edit their
        contents: e.g. ``snapshot.particles.typeid[:] == 0``.

    Warning:
        Data is only present on the root rank:

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                pos = snapshot.particles.position[0]

    See Also:
        `State`

        `Simulation.create_state_from_snapshot`

        `State.get_snapshot`

        `State.set_snapshot`

    .. rubric:: Example:

    .. code-block:: python

        snapshot = hoomd.Snapshot()

    Attributes:
        communicator (Communicator): MPI communicator.
    """

    def __init__(self, communicator=None):
        if communicator is None:
            self.communicator = hoomd.communicator.Communicator()
        else:
            self.communicator = communicator

        self._cpp_obj = _hoomd.SnapshotSystemData_double()

    @property
    def configuration(self):
        """Snapshot box configuration.

        Attributes:
            dimensions (int): Number of dimensions
            box (tuple[float, float, float, float, float, float]): Simulation
              box parameters ``[Lx, Ly, Lz, xy, xz, yz]``.

        Note:
            ``box`` accepts any values that `Box.from_box` allows when setting.

        See Also:
            `Box`

        .. rubric:: Example:

        .. code-block:: python

            snapshot.configuration.box = [10, 20, 30, 0.1, 0.2, 0.3]
        """
        return _ConfigurationData(self._cpp_obj)

    @property
    def particles(self):
        """Particles.

        Attributes:
            particles.N (int): Number of particles in the snapshot.

            particles.types (list[str]):
                Names of the particle types.

            particles.position ((*N*, 3) `numpy.ndarray` of `float`):
                Particle position :math:`[\\mathrm{length}]`.

            particles.orientation ((*N*, 4) `numpy.ndarray` of \
                `float`):
                Particle orientation.

            particles.typeid ((*N*, ) `numpy.ndarray` of ``uint32``):
                Particle type id.

            particles.mass ((*N*, ) `numpy.ndarray` of `float`):
                Particle mass :math:`[\\mathrm{mass}]`.

            particles.charge ((*N*, ) `numpy.ndarray` of `float`):
                Particle charge :math:`[\\mathrm{charge}]`.

            particles.diameter ((*N*, ) `numpy.ndarray` of `float`):
                Particle diameter :math:`[\\mathrm{length}]`.

            particles.body ((*N*, ) `numpy.ndarray` of ``int32``):
                Particle body.

            particles.moment_inertia ((*N*, 3) `numpy.ndarray` of \
                `float`):
                Particle moment of inertia :math:`[\\mathrm{mass} \\cdot
                \\mathrm{length}^2]`.

            particles.velocity ((*N*, 3) `numpy.ndarray` of `float`):
                Particle velocity :math:`[\\mathrm{velocity}]`.

            particles.angmom ((*N*, 4) `numpy.ndarray` of `float`):
                Particle angular momentum :math:`[\\mathrm{mass} \\cdot
                \\mathrm{velocity} \\cdot \\mathrm{length}]`.

            particles.image ((*N*, 3) `numpy.ndarray` of ``int32``):
                Particle image.

        Note:
            Set ``N`` to change the size of the arrays.

        .. rubric:: Example:

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                snapshot.particles.N = 4
                snapshot.particles.position[:] = [[0, 0, 0],
                                                [1, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 1]]
                snapshot.particles.types = ['A', 'B']
                snapshot.particles.typeid[:] = [0, 1, 1, 0]
        """
        if self.communicator.rank == 0:
            return self._cpp_obj.particles
        else:
            raise RuntimeError('Snapshot data is only present on rank 0')

    @property
    def bonds(self):
        """Bonds.

        Attributes:
            bonds.N (int): Number of bonds.

            bonds.types (list[str]): Names of the bond types

            bonds.typeid ((*N*,) `numpy.ndarray` of ``uint32``):
                Bond type id.

            bonds.group ((*N*, 2) `numpy.ndarray` of ``uint32``):
                Tags of the particles in the bond.

        Note:
            Set ``N`` to change the size of the arrays.

        .. rubric:: Example:

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                snapshot.bonds.N = 2
                snapshot.bonds.group[:] = [[0, 1], [2, 3]]
                snapshot.bonds.types = ['A-B']
                snapshot.bonds.typeid[:] = [0, 0]
        """
        if self.communicator.rank == 0:
            return self._cpp_obj.bonds
        else:
            raise RuntimeError('Snapshot data is only present on rank 0')

    @property
    def angles(self):
        """Angles.

        Attributes:
            angles.N (int): Number of angles.

            angles.types (list[str]): Names of the angle types

            angles.typeid ((*N*,) `numpy.ndarray` of ``uint32``):
                Angle type id.

            angles.group ((*N*, 3) `numpy.ndarray` of ``uint32``):
                Tags of the particles in the angle.

        Note:
            Set ``N`` to change the size of the arrays.

        .. rubric:: Example:

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                snapshot.angles.N = 1
                snapshot.angles.group[:] = [[0, 1, 2]]
                snapshot.angles.types = ['A-B-B']
                snapshot.angles.typeid[:] = [0]
        """
        if self.communicator.rank == 0:
            return self._cpp_obj.angles
        else:
            raise RuntimeError('Snapshot data is only present on rank 0')

    @property
    def dihedrals(self):
        """Dihedrals.

        Attributes:
            dihedrals.N (int): Number of dihedrals.

            dihedrals.types (list[str]): Names of the dihedral types

            dihedrals.typeid ((*N*,) `numpy.ndarray` of ``uint32``):
                Dihedral type id.

            dihedrals.group ((*N*, 4) `numpy.ndarray` of ``uint32``):
                Tags of the particles in the dihedral.

        Note:
            Set ``N`` to change the size of the arrays.

        .. rubric:: Example:

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                snapshot.dihedrals.N = 1
                snapshot.dihedrals.group[:] = [[0, 1, 2, 3]]
                snapshot.dihedrals.types = ['A-B-B-A']
                snapshot.dihedrals.typeid[:] = [0]
        """
        if self.communicator.rank == 0:
            return self._cpp_obj.dihedrals
        else:
            raise RuntimeError('Snapshot data is only present on rank 0')

    @property
    def impropers(self):
        """Impropers.

        Attributes:
            impropers.N (int): Number of impropers.

            impropers.types (list[str]): Names of the improper types

            impropers.typeid ((*N*,) `numpy.ndarray` of ``uint32``):
                Improper type id.

            impropers.group ((*N*, 4) `numpy.ndarray` of ``uint32``):
                Tags of the particles in the improper.

        Note:
            Set ``N`` to change the size of the arrays.

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                snapshot.impropers.N = 1
                snapshot.impropers.group[:] = [[0, 1, 2, 3]]
                snapshot.impropers.types = ['A-B-B-A']
                snapshot.impropers.typeid[:] = [0]
        """
        if self.communicator.rank == 0:
            return self._cpp_obj.impropers
        else:
            raise RuntimeError('Snapshot data is only present on rank 0')

    @property
    def pairs(self):
        """Special pairs.

        Attributes:
            pairs.N (int): Number of special pairs.

            pairs.types (list[str]): Names of the special pair types

            pairs.typeid ((*N*,) `numpy.ndarray` of ``uint32``):
                Special pair type id.

            pairs.group ((*N*, 2) `numpy.ndarray` of ``uint32``):
                Tags of the particles in the special pair.

        Note:
            Set ``N`` to change the size of the arrays.

        .. rubric:: Example:

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                snapshot.pairs.N = 2
                snapshot.pairs.group[:] = [[0, 1], [2, 3]]
                snapshot.pairs.types = ['A-B']
                snapshot.pairs.typeid[:] = [0, 0]
        """
        if self.communicator.rank == 0:
            return self._cpp_obj.pairs
        else:
            raise RuntimeError('Snapshot data is only present on rank 0')

    @property
    def constraints(self):
        """Constraints.

        Attributes:
            constraints.N (int): Number of constraints.

            constraints.value ((*N*, ) `numpy.ndarray` of `float`):
                Constraint length.

            constraints.group ((*N*, *2*) `numpy.ndarray` of ``uint32``):
                Tags of the particles in the constraint.

        Note:
            Set ``N`` to change the size of the arrays.

        .. rubric:: Example:

        .. code-block:: python

            if snapshot.communicator.rank == 0:
                snapshot.constraints.N = 2
                snapshot.constraints.group[:] = [[0, 1], [2, 3]]
                snapshot.constraints.value[:] = [1, 1]
        """
        if self.communicator.rank == 0:
            return self._cpp_obj.constraints
        else:
            raise RuntimeError('Snapshot data is only present on rank 0')

    @property
    def mpcd(self):
        """MPCD data.

        Attributes:
            mpcd.N (int): Number of MPCD particles.

            mpcd.position ((*N*, 3) `numpy.ndarray` of `float`):
                Particle position :math:`[\\mathrm{length}]`.

            mpcd.velocity ((*N*, 3) `numpy.ndarray` of `float`):
                Particle velocity :math:`[\\mathrm{velocity}]`.

            mpcd.types (list[str]):
                Names of the particle types.

            mpcd.typeid ((*N*, ) `numpy.ndarray` of ``uint32``):
                Particle type id.

            mpcd.mass (float): Particle mass.

        Note:
            Set ``N`` to change the size of the arrays.

        Note:
            This attribute is only available when
            HOOMD-blue is built with the MPCD component.
        """
        if not hoomd.version.mpcd_built:
            raise RuntimeError("MPCD component not built")

        if self.communicator.rank == 0:
            return self._cpp_obj.mpcd
        else:
            raise RuntimeError("Snapshot data is only present on rank 0")

    @classmethod
    def _from_cpp_snapshot(cls, snapshot, communicator):
        sp = cls()
        sp.communicator = communicator
        sp._cpp_obj = snapshot
        return sp

    def replicate(self, nx, ny, nz=1):
        """Replicate the snapshot along the periodic box directions.

        Args:
            nx (int): Number of times to replicate in the x direction.
            ny (int): Number of times to replicate in the y direction.
            nz (int): Number of times to replicate in the z direction.

        Performs the same operation as `State.replicate` on a `Snapshot`.

        Returns:
            ``self``

        .. rubric:: Example:

        .. code-block:: python

            snapshot.replicate(nx=2, ny=2, nz=2)
        """
        self._cpp_obj.replicate(nx, ny, nz)
        return self

    def wrap(self):
        """Wrap particles into the snapshot box.

        Returns:
            ``self``

        .. rubric:: Example:

        .. code-block:: python

            snapshot.wrap()
        """
        self._cpp_obj.wrap()
        return self

    def _broadcast_box(self):
        self._cpp_obj._broadcast_box(self.communicator.cpp_mpi_conf)

    @classmethod
    def from_gsd_frame(cls, gsd_snap, communicator):
        """Constructs a `hoomd.Snapshot` from a `gsd.hoomd.Frame` object.

        Args:
            gsd_snap (gsd.hoomd.Frame): The gsd frame to convert to a
                `hoomd.Snapshot`.
            communicator (hoomd.communicator.Communicator): The MPI communicator
                to use for the snapshot. This prevents the snapshot from being
                stored on every rank.

        Tip:
            Use `Simulation.create_state_from_gsd` to efficiently initialize
            the system state from a GSD file.

        Note:
            `from_gsd_frame` only accesses the ``gsd_snap`` argument on rank
            0. In MPI simulations, avoid duplicating memory and file reads by
            reading GSD files only on rank 0 and passing ``gsd_snap=None`` on
            other ranks.
        """
        snap = cls(communicator=communicator)

        def set_properties(snap_section, gsd_snap_section, properties,
                           array_properties):
            for prop in properties:
                gsd_prop = getattr(gsd_snap_section, prop, None)
                if gsd_prop is not None:
                    setattr(snap_section, prop, gsd_prop)
            for prop in array_properties:
                gsd_prop = getattr(gsd_snap_section, prop, None)
                if gsd_prop is not None:
                    getattr(snap_section, prop)[:] = gsd_prop

        if communicator.rank == 0:

            gsd_snap.validate()

            set_properties(snap.particles, gsd_snap.particles, ('N', 'types'),
                           ('angmom', 'body', 'charge', 'diameter', 'image',
                            'mass', 'moment_inertia', 'orientation', 'position',
                            'typeid', 'velocity'))

            for section in ('angles', 'bonds', 'dihedrals', 'impropers',
                            'pairs'):
                set_properties(getattr(snap,
                                       section), getattr(gsd_snap, section),
                               ('N', 'types'), ('group', 'typeid'))

            set_properties(snap.constraints, gsd_snap.constraints, ('N',),
                           ('group', 'value'))

            # Set box attribute
            if gsd_snap.configuration.box is not None:
                box = list(gsd_snap.configuration.box)
                if gsd_snap.configuration.dimensions == 2:
                    box[2] = 0
                snap.configuration.box = box

        snap._broadcast_box()
        return snap

    @classmethod
    def from_gsd_snapshot(cls, gsd_snap, communicator):
        """Constructs a `hoomd.Snapshot` from a ``gsd.hoomd.Snapshot`` object.

        .. deprecated:: 4.0.0

            Use `from_gsd_frame`.
        """
        warnings.warn("gsd.hoomd.Snapshot is deprecated, use gsd.hoomd.Frame",
                      FutureWarning)
        return cls.from_gsd_frame(gsd_snap, communicator)
