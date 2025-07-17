# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Access simulation state data directly."""

from abc import ABC, abstractmethod
from hoomd import Box
from hoomd import _hoomd


class _LocalAccess(ABC):
    __slots__ = ("_accessed_fields", "_cpp_obj", "_entered")
    _global_fields = {"rtag": "getRTags"}

    @property
    @abstractmethod
    def _fields(self):
        pass

    @property
    @abstractmethod
    def _array_cls(self):
        pass

    def __init__(self):
        self._entered = False
        self._accessed_fields = dict()

    def __getattr__(self, attr):
        if attr in self._accessed_fields:
            return self._accessed_fields[attr]
        elif attr in self._global_fields:
            buff = getattr(self._cpp_obj, self._global_fields[attr])()
        else:
            raw_attr, flag = self._get_raw_attr_and_flag(attr)
            if raw_attr in self._fields:
                buff = getattr(self._cpp_obj, self._fields[raw_attr])(flag)
            else:
                raise AttributeError(
                    "{} object has no attribute {}".format(type(self), attr)
                )

        self._accessed_fields[attr] = arr = self._array_cls(buff, lambda: self._entered)
        return arr

    def _get_raw_attr_and_flag(self, attr):
        ghosts_only = attr.startswith("ghost_")
        with_ghosts = attr.endswith("_with_ghost")
        raw_attr = attr.replace("_with_ghost", "").replace("ghost_", "")
        if ghosts_only and with_ghosts:
            raise ValueError(
                "Attribute cannot be both prefixed with ghost_ "
                "and suffixed with _with_ghost"
            )
        elif ghosts_only:
            return raw_attr, _hoomd.GhostDataFlag.ghost
        elif with_ghosts:
            return raw_attr, _hoomd.GhostDataFlag.both
        else:
            return raw_attr, _hoomd.GhostDataFlag.standard

    def __setattr__(self, attr, value):
        if attr in self.__slots__:
            super().__setattr__(attr, value)
            return
        try:
            arr = getattr(self, attr)
        except AttributeError:
            raise AttributeError(
                "{} object has no attribute {}.".format(self.__class__, attr)
            )
        else:
            if arr.read_only:
                raise RuntimeError("Attribute {} is not settable.".format(attr))
            arr[:] = value

    def _enter(self):
        self._cpp_obj.enter()
        self._entered = True

    def _exit(self):
        self._cpp_obj.exit()
        self._entered = False
        self._accessed_fields = dict()


class ParticleLocalAccessBase(_LocalAccess):
    """Directly access particle data in the simulation state.

    Note:
        Changing some attributes (such as ``velocity`` and ``acceleration``)
        may not alter the trajectory of the system as you would expect.
        The `md.Integrator` is responsible for integrating the equations of
        motion and manages the values in these arrays.

    See Also:
        * `hoomd.State`
        * `hoomd.data.LocalSnapshot`
        * `hoomd.data.LocalSnapshotGPU`

    Attributes:
        typeid ((N_particles) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            The integer type of a particle.
        tag ((N_particles) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The particle tags. Spatial sorting and MPI domain migration
            reorders particles in memory. The particle tag identifies each
            particle in the order it existed in the initial configuration.
        rtag ((N_particles_global) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The particle reverse tags. For a given particle tag ``tag``,
            ``i = particles.rtag[tag]`` is the array index holding that
            particle.
        position ((N_particles, 3) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Particle positions :math:`[\\mathrm{length}]`.
        image ((N_particles, 3) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            A count of how many times each particle crosses the periodic box
            boundaries.
        velocity ((N_particles, 3) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Particle velocities :math:`[\\mathrm{velocity}]`.
        acceleration ((N_particles, 3) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Particle accelerations
            :math:`[\\mathrm{velocity} \\cdot \\mathrm{time}^{-1}]`.
        mass ((N_particles) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Particle masses :math:`[\\mathrm{mass}]`.
        orientation ((N_particles, 4) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Particle orientations expressed as quaternions.
        angmom ((N_particles, 4) `HOOMDArray` or `HOOMDGPUArray` of \
            ``float``):
            Particle angular momenta expressed as quaternions
            :math:`[\\mathrm{mass} \\cdot \\mathrm{velocity} \\cdot
            \\mathrm{length}]`.
        moment_inertia ((N_particles, 3) `HOOMDArray` or `HOOMDGPUArray` of \
            ``float``):
            Particle principal moments of inertia
            :math:`[\\mathrm{mass} \\cdot \\mathrm{length}^2]`.
        charge ((N_particles) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Particle electrical charges :math:`[\\mathrm{charge}]`.
        diameter ((N_particles) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Particle diameters :math:`[\\mathrm{length}]`.
        body ((N_particles) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The id of the rigid body the particle is in.
        net_force ((N_particles, 3) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Net force on particle :math:`[\\mathrm{force}]`.
        net_torque ((N_particles, 3) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Net torque on particle
            :math:`[\\mathrm{force} \\cdot \\mathrm{length}]`.
        net_virial ((N_particles, 6) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Net virial on particle :math:`[\\mathrm{energy}]`.
        net_energy ((N_particles,) `HOOMDArray` or `HOOMDGPUArray` of ``float``):
            Net energy of a particle :math:`[\\mathrm{energy}]`.
    """

    @property
    @abstractmethod
    def _cpp_cls(self):
        pass

    _fields = {
        "position": "getPosition",
        "typeid": "getTypes",
        "velocity": "getVelocities",
        "mass": "getMasses",
        "acceleration": "getAcceleration",
        "orientation": "getOrientation",
        "angmom": "getAngularMomentum",
        "moment_inertia": "getMomentsOfInertia",
        "charge": "getCharge",
        "diameter": "getDiameter",
        "image": "getImages",
        "tag": "getTags",
        "rtag": "getRTags",
        "body": "getBodies",
        "net_force": "getNetForce",
        "net_torque": "getNetTorque",
        "net_virial": "getNetVirial",
        "net_energy": "getNetEnergy",
    }

    def __init__(self, state):
        super().__init__()
        self._cpp_obj = self._cpp_cls(state._cpp_sys_def.getParticleData())


class _GroupLocalAccess(_LocalAccess):
    @property
    @abstractmethod
    def _cpp_cls(self):
        pass

    @property
    @abstractmethod
    def _cpp_get_data_method_name(self):
        pass

    _fields = {
        "typeid": "getTypeVal",
        "group": "getMembers",
        "tag": "getTags",
        "rtag": "getRTags",
    }

    def __init__(self, state):
        super().__init__()
        self._cpp_obj = self._cpp_cls(
            getattr(state._cpp_sys_def, self._cpp_get_data_method_name)()
        )


class BondLocalAccessBase(_GroupLocalAccess):
    """Directly access bond data in the simulation state.

    See Also:
        * `hoomd.State`
        * `hoomd.data.LocalSnapshot`
        * `hoomd.data.LocalSnapshotGPU`

    Attributes:
        typeid ((N_bonds) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The integer type of a bond.
        members ((N_bonds, 2) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The tags of particles in a bond.
        tag ((N_bonds) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The bond tags. MPI domain migration reorders bonds in memory. The
            bond tag identifies each bond in the order it existed in the initial
            configuration.
        rtag ((N_bonds_global) `HOOMDArray` or `HOOMDGPUArray` of ``int``): the
            The bond reverse tags. For a given bond tag ``tag``,
            ``i = bonds.rtag[tag]`` is the array index holding that
            bond.
    """

    _cpp_get_data_method_name = "getBondData"


class AngleLocalAccessBase(_GroupLocalAccess):
    """Directly access angle data in the simulation state.

    See Also:
        * `hoomd.State`
        * `hoomd.data.LocalSnapshot`
        * `hoomd.data.LocalSnapshotGPU`

    Attributes:
        typeid ((N_angles) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The integer type of a angle.
        members ((N_angles, 3) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The tags of particles in a angle.
        tag ((N_angles) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The angle tags. MPI domain migration reorders angles in memory.
            The angle tag identifies each angle in the order it existed in the
            initial configuration.
        rtag ((N_angles_global) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The angle reverse tags. For a given angle tag ``tag``, ``i =
            angles.rtag[tag]`` is the array index holding that angle.
    """

    _cpp_get_data_method_name = "getAngleData"


class DihedralLocalAccessBase(_GroupLocalAccess):
    """Direct access dihedral data in the simulation state.

    See Also:
        * `hoomd.State`
        * `hoomd.data.LocalSnapshot`
        * `hoomd.data.LocalSnapshotGPU`

    Attributes:
        typeid ((N_dihedrals) `HOOMDArray` or `HOOMDGPUArray` of ``int``): The integer
            type of a dihedral.
        members ((N_dihedrals, 4) `HOOMDArray` or `HOOMDGPUArray` of ``int``): the
            tags of particles in a dihedral.
        tag ((N_dihedrals) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The dihedral tags. MPI domain migration reorders dihedrals in
            memory. The dihedral tag identifies each dihedral in the order it
            existed in the initial configuration.
        rtag ((N_dihedrals_global) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The dihedral reverse tags. For a given dihedral tag ``tag``, ``i
            = dihedrals.rtag[tag]`` is the array index holding that dihedral.
    """

    _cpp_get_data_method_name = "getDihedralData"


class ImproperLocalAccessBase(_GroupLocalAccess):
    """Directly access improper data in the simulation state.

    See Also:
        * `hoomd.State`
        * `hoomd.data.LocalSnapshot`
        * `hoomd.data.LocalSnapshotGPU`

    Attributes:
        typeid ((N_impropers) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The integer type of a improper.
        members ((N_impropers, 3) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The tags of particles in a improper.
        tag ((N_impropers) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The improper tags. MPI domain migration reorders impropers in
            memory. The improper tag identifies each improper in the order it
            existed in the initial configuration.
        rtag ((N_impropers_global) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The improper reverse tags. For a given improper tag ``tag``, ``i
            = impropers.rtag[tag]`` is the array index holding that improper.
    """

    _cpp_get_data_method_name = "getImproperData"


class ConstraintLocalAccessBase(_GroupLocalAccess):
    """Directly access constraint data in the simulation state.

    See Also:
        * `hoomd.State`
        * `hoomd.data.LocalSnapshot`
        * `hoomd.data.LocalSnapshotGPU`

    Attributes:
        value ((N_constraints) `HOOMDArray` or `HOOMDGPUArray` of ``float``): The
            constraint value.
        members ((N_constraints, 3) `HOOMDArray` or `HOOMDGPUArray` of ``int``): the
            tags of particles in a constraint.
        tag ((N_constraints) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The constraint tags. MPI domain migration reorders constraints in
            memory. The constraint tag identifies each constraint in the order
            it existed in the initial configuration.
        rtag ((N_constraints_global) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The constraint reverse tags. For a given constraint tag ``tag``,
            ``i = constraints.rtag[tag]`` is the array index holding that
            constraint.
    """

    _fields = {
        "value": "getTypeVal",
        "group": "getMembers",
        "tag": "getTags",
        "rtag": "getRTags",
    }
    _cpp_get_data_method_name = "getConstraintData"


class PairLocalAccessBase(_GroupLocalAccess):
    """Directly access special pair data in the simulation state.

    See Also:
        * `hoomd.State`
        * `hoomd.data.LocalSnapshot`
        * `hoomd.data.LocalSnapshotGPU`

    Attributes:
        typeid ((N_pairs) `HOOMDArray` or `HOOMDGPUArray` of ``float``): The type of
            special pair.
        members ((N_pairs, 3) `HOOMDArray` or `HOOMDGPUArray` of ``int``): the tags of
            particles in a special pair.
        tag ((N_special_pairs) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The special pair tags. MPI domain migration reorders special
            pairs in memory. The special pair tag identifies each special pair
            in the order it existed in the initial configuration.
        rtag ((N_special_pairs_global) `HOOMDArray` or `HOOMDGPUArray` of ``int``):
            The special pair reverse tags. For a given special pair tag
            ``tag``, ``i = pairs.rtag[tag]`` is the array index holding that
            special pair.
    """

    _cpp_get_data_method_name = "getPairData"


class _LocalSnapshot:
    def __init__(self, state):
        self._state = state
        self._box = state.box
        self._local_box = state._cpp_sys_def.getParticleData().getBox()

    @property
    def global_box(self):
        """hoomd.Box: The global simulation box."""
        return Box.from_box(self._box)

    @property
    def local_box(self):
        """hoomd.Box: The local box according to the domain decomposition."""
        return Box.from_box(Box._from_cpp(self._local_box))

    @property
    def particles(self):
        """hoomd.data.ParticleLocalAccessBase: Local particle data."""
        return self._particles

    @property
    def bonds(self):
        """hoomd.data.BondLocalAccessBase: Local bond data."""
        return self._bonds

    @property
    def angles(self):
        """hoomd.data.AngleLocalAccessBase: Local angle data."""
        return self._angles

    @property
    def dihedrals(self):
        """hoomd.data.DihedralLocalAccessBase: Local dihedral data."""
        return self._dihedrals

    @property
    def impropers(self):
        """hoomd.data.ImproperLocalAccessBase: Local improper data."""
        return self._impropers

    @property
    def constraints(self):
        """hoomd.data.ConstraintLocalAccessBase: Local constraint data."""
        return self._constraints

    @property
    def pairs(self):
        """hoomd.data.PairLocalAccessBase: Local special pair data."""
        return self._pairs

    def __enter__(self):
        self._state._in_context_manager = True
        self._particles._enter()
        self._bonds._enter()
        self._angles._enter()
        self._dihedrals._enter()
        self._impropers._enter()
        self._constraints._enter()
        self._pairs._enter()
        return self

    def __exit__(self, type, value, traceback):
        self._state._in_context_manager = False
        self._particles._exit()
        self._bonds._exit()
        self._angles._exit()
        self._dihedrals._exit()
        self._impropers._exit()
        self._constraints._exit()
        self._pairs._exit()
