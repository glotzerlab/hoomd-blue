from abc import ABC, abstractmethod
from hoomd import Box
from hoomd.array import HOOMDArray
from hoomd import _hoomd


class _LocalAccess(ABC):
    @property
    @abstractmethod
    def _fields(self):
        pass

    @property
    @abstractmethod
    def _array_cls(self):
        pass

    def __init__(self):
        object.__setattr__(self, '_entered', False)
        object.__setattr__(self, '_accessed_fields', dict())

    def __getattr__(self, attr):
        if attr in self._accessed_fields:
            return self._accessed_fields[attr]
        elif attr.startswith("ghost_"):
            if attr[6:] in self._fields:
                buff = getattr(self._cpp_obj, self._fields[attr[6:]])(
                    True, False)
            else:
                raise AttributeError(
                    "{} object has no attribute {}".format(type(self), attr))
        elif attr.endswith("_with_ghosts"):
            if attr[:-12] in self._fields:
                buff = getattr(self._cpp_obj, self._fields[attr[:-12]])(
                    False, True)
            else:
                raise AttributeError(
                    "{} object has no attribute {}".format(type(self), attr))
        elif attr in self._fields:
            buff = getattr(self._cpp_obj, self._fields[attr])(False, False)
        else:
            raise AttributeError(
                "{} object has no attribute {}".format(type(self), attr))
        arr = self._array_cls(buff, lambda: self._entered)
        self._accessed_fields[attr] = arr
        return arr

    def __setattr__(self, attr, value):
        try:
            arr = getattr(self, attr)
        except AttributeError:
            raise AttributeError(
                "Cannot set attribute {}, does not exist.".format(attr))
        if arr.read_only == True:
            raise RuntimeError(
                "Attribute {} is not settable.".format(attr))
        arr[:] = value

    def _enter(self):
        self._cpp_obj.enter()
        object.__setattr__(self, '_entered', True)

    def _exit(self):
        self._cpp_obj.exit()
        object.__setattr__(self, '_entered', False)
        object.__setattr__(self, '_accessed_fields', dict())


class _ParticleLocalAccess(_LocalAccess):
    @property
    @abstractmethod
    def _cpp_cls(self):
        pass

    _fields = {
        'position': 'getPosition',
        'typeid': 'getTypes',
        'velocity': 'getVelocities',
        'mass': 'getMasses',
        'acceleration': 'getAcceleration',
        'orientation': 'getOrientation',
        'angular_momentum': 'getAngularMomentum',
        'moment_of_inertia': 'getMomentsOfInertia',
        'charge': 'getCharge',
        'diameter': 'getDiameter',
        'image': 'getImages',
        'tag': 'getTags',
        'rtag': 'getRTags',
        'rigid_body_id': 'getBodies',
        'net_force': 'getNetForce',
        'net_torque': 'getNetTorque',
        'net_virial': 'getNetVirial'}

    def __init__(self, state):
        super().__init__()
        object.__setattr__(
            self, '_cpp_obj',
            self._cpp_cls(state._cpp_sys_def.getParticleData()))


class ParticleLocalAccessCPU(_ParticleLocalAccess):
    _cpp_cls = _hoomd.LocalParticleDataHost
    _array_cls = HOOMDArray


class _GroupLocalAccess(_LocalAccess):
    @property
    @abstractmethod
    def _cpp_cls(self):
        pass

    @property
    @abstractmethod
    def _cpp_data_get_method(self):
        pass

    _fields = {
        'typeid': 'getTypeVal',
        'group': 'getMembers',
        'tag': 'getTags',
        'rtag': 'getRTags'
    }

    def __init__(self, state):
        super().__init__()
        object.__setattr__(
            self, '_cpp_obj',
            self._cpp_cls(
                getattr(state._cpp_sys_def, self._cpp_data_get_method)()
            )
        )


class BondLocalAccessCPU(_GroupLocalAccess):
    _cpp_cls = _hoomd.LocalBondDataHost
    _cpp_data_get_method = "getBondData"
    _array_cls = HOOMDArray


class AngleLocalAccessCPU(_GroupLocalAccess):
    _cpp_cls = _hoomd.LocalAngleDataHost
    _cpp_data_get_method = "getAngleData"
    _array_cls = HOOMDArray


class DihedralLocalAccessCPU(_GroupLocalAccess):
    _cpp_cls = _hoomd.LocalDihedralDataHost
    _cpp_data_get_method = "getDihedralData"
    _array_cls = HOOMDArray


class ImproperLocalAccessCPU(_GroupLocalAccess):
    _cpp_cls = _hoomd.LocalImproperDataHost
    _cpp_data_get_method = "getImproperData"
    _array_cls = HOOMDArray


class ConstraintLocalAccessCPU(_GroupLocalAccess):
    _fields = {
        'value': 'getTypeVal',
        'group': 'getMembers',
        'tag': 'getTags',
        'rtag': 'getRTags'
    }
    _cpp_cls = _hoomd.LocalConstraintDataHost
    _cpp_data_get_method = "getConstraintData"
    _array_cls = HOOMDArray


class PairLocalAccessCPU(_GroupLocalAccess):
    _cpp_cls = _hoomd.LocalPairDataHost
    _cpp_data_get_method = "getPairData"
    _array_cls = HOOMDArray


class _LocalSnapshotBase:
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
        """Local particle data.

        Attributes:
            type ((N) `hoomd.array` object of ``float``): The integer type of a
                particle
            tag ((N) `hoomd.array` object of ``int``): the tag of a particle.
                HOOMD-blue uses spacial sorting to improve cache efficiency in
                particle look-ups. This means the ordering of the array changes.
                However, particle tags remain constant. This means that if
                ``particles.tag[0]`` is 1, then later whatever index has a value
                of 1 later in the simulation is the same particle.
            rtag ((N) `hoomd.array` object of ``int``): the reverse tag of a
                particle. This means that the value ``particles.rtag[0]``
                represents the current index for particle with tag 0.
            position ((N, 3) `hoomd.array` object of ``float``): particle
                positions
            image ((N, 3) `hoomd.array` object of ``int``): the periodic image
                a particle occupies
            velocity ((N, 3) `hoomd.array` object of ``float``): particle
                velocities (note that changing velocities may or may not effect
                the future trajectory)
            acceleration ((N, 3) `hoomd.array` object of ``float``): particle
                accelerations (note that changing accelerations may or may not
                effect the future trajectory)
            mass ((N) `hoomd.array` object of ``float``): particles' masses
            orientation ((N, 4) `hoomd.array` object of ``float``): particle
                orientations expressed as quaternions
            angular_momentum ((N, 4) `hoomd.array` object of ``float``):
                particle angular momenta expressed as quaternions
            moment_of_inertia ((N, 3) `hoomd.array` object of ``float``):
                particle principle moments of inertia
            charge ((N) `hoomd.array` object of ``float``): particle electrical
                charges
            diameter ((N) `hoomd.array` object of ``float``): particle
                diameters
            rigid_body_id ((N) `hoomd.array` object of ``int``): The id of the
                rigid body the particle is in.
            net_force ((N, 3) `hoomd.array` object of ``float``): net force on
                particle
            net_torque ((N, 3) `hoomd.array` object of ``float``): net torque
                on particle
            net_virial ((N, 3) `hoomd.array` object of ``float``): net virial
                on particle
        """
        return self._particles

    @property
    def bonds(self):
        """Local bond data.

        Attributes:
            typeid ((N_bonds) `hoomd.array` object of ``int``): The integer
                type of a bond.
            members ((N_bonds, 2) `hoomd.array` object of ``int``): the tags of
                particles in a bond.
            tag ((N_bonds) `hoomd.array` object of ``int``): the tag of the
                bond.  HOOMD-blue uses spacial sorting to improve cache
                efficiency in bond look-ups. This means the ordering of the
                array changes.  However, bond tags remain constant. This means
                that if ``bond.tag[0]`` is 1, then later whatever index has a
                value of 1 later in the simulation is the same bond.
            rtag ((N_bonds) `hoomd.array` object of ``int``): the reverse tag
                of a bond. This means that the value ``bond.rtag[0]`` represents
                the current index for bond with tag 0.
        """
        return self._bonds

    @property
    def angles(self):
        """Local angle data.

        Attributes:
            typeid ((N_angles) `hoomd.array` object of ``int``): The integer
                type of a angle.
            members ((N_angles, 3) `hoomd.array` object of ``int``): the tags
                of particles in a angle.
            tag ((N_angles) `hoomd.array` object of ``int``): the tag of the
                angle.  HOOMD-blue uses spacial sorting to improve cache
                efficiency in angle look-ups. This means the ordering of the
                array changes.  However, angle tags remain constant. This means
                that if ``angle.tag[0]`` is 1, then later whatever index has a
                value of 1 later in the simulation is the same angle.
            rtag ((N_angles) `hoomd.array` object of ``int``): the reverse tag
                of a angle. This means that the value ``angle.rtag[0]``
                represents the current index for angle with tag 0.
        """
        return self._angles

    @property
    def dihedrals(self):
        """Local dihedral data.

        Attributes:
            typeid ((N_dihedrals) `hoomd.array` object of ``int``): The integer
                type of a dihedral.
            members ((N_dihedrals, 3) `hoomd.array` object of ``int``): the
                tags of particles in a dihedral.
            tag ((N_dihedrals) `hoomd.array` object of ``int``): the tag of the
                dihedral.  HOOMD-blue uses spacial sorting to improve cache
                efficiency in dihedral look-ups. This means the ordering of the
                array changes.  However, dihedral tags remain constant. This
                means that if ``dihedral.tag[0]`` is 1, then later whatever
                index has a value of 1 later in the simulation is the same
                dihedral.
            rtag ((N_dihedrals) `hoomd.array` object of ``int``): the reverse
                tag of a dihedral. This means that the value
                ``dihedral.rtag[0]`` represents the current index for dihedral
                with tag 0.
        """
        return self._dihedrals

    @property
    def impropers(self):
        """Local improper data.

        Attributes:
            typeid ((N_impropers) `hoomd.array` object of ``int``): The integer
                type of a improper.
            members ((N_impropers, 3) `hoomd.array` object of ``int``): the
                tags of particles in a improper.
            tag ((N_impropers) `hoomd.array` object of ``int``): the tag of the
                improper.  HOOMD-blue uses spacial sorting to improve cache
                efficiency in improper look-ups. This means the ordering of the
                array changes.  However, improper tags remain constant. This
                means that if ``improper.tag[0]`` is 1, then later whatever
                index has a value of 1 later in the simulation is the same
                improper.
            rtag ((N_impropers) `hoomd.array` object of ``int``): the reverse
                tag of a improper. This means that the value
                ``improper.rtag[0]`` represents the current index for improper
                with tag 0.
        """
        return self._impropers

    @property
    def constraints(self):
        """Local constraint data.

        Attributes:
            value ((N_constraints) `hoomd.array` object of ``float``): The
                constaint value.
            members ((N_constraints, 3) `hoomd.array` object of ``int``): the
                tags of particles in a constraint.
            tag ((N_constraints) `hoomd.array` object of ``int``): the tag of
                the constraint.  HOOMD-blue uses spacial sorting to improve
                cache efficiency in constraint look-ups. This means the ordering
                of the array changes.  However, constraint tags remain constant.
                This means that if ``constraint.tag[0]`` is 1, then later
                whatever index has a value of 1 later in the simulation is the
                same constraint.
            rtag ((N_constraints) `hoomd.array` object of ``int``): the reverse
                tag of a constraint. This means that the value
                ``constraint.rtag[0]`` represents the current index for
                constraint with tag 0.
        """
        return self._constraints

    @property
    def pairs(self):
        """Local special pair data.

        Attributes:
            typeid ((N_pairs) `hoomd.array` object of ``float``): The type of
                special pair.
            members ((N_pairs, 3) `hoomd.array` object of ``int``): the tags of
                particles in a special pair.
            tag ((N_pairs) `hoomd.array` object of ``int``): the tag of the
                special pair.  HOOMD-blue uses spacial sorting to improve cache
                efficiency in pair look-ups. This means the ordering of the
                array changes.  However, special pair tags remain constant.
                This means that if ``pair.tag[0]`` is 1, then later whatever
                index has a value of 1 later in the simulation is the same pair.
            rtag ((N_pairs) `hoomd.array` object of ``int``): the reverse tag
                of a special pair. This means that the value ``pair.rtag[0]``
                represents the current index for special pair with tag 0.
        """
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


class LocalSnapshot(_LocalSnapshotBase):
    """Provides context manager access to HOOMD-blue CPU data buffers.

    The interface of a `LocalSnapshot` is similar to that of the
    `hoomd.Snapshot`. Data is MPI rank local so for MPI parallel simulations
    only the data possessed by a rank is exposed. This means that users must
    handle the domain decomposition directly. One consequence of this is that
    access to ghost particle data is provided. A ghost particle is a particle
    that is not owned by a rank, but nevertheless is required for operations
    that use particle neighbors. Also, changing the global or local box within a
    `LocalSnapshot` context manager is not allowed.

    For every property (e.g. ``data.particles.position``), only grabs the
    data for the regular (non-ghost) particles. The property can be prefixed
    with ``ghost_`` to grab the ghost particles in a read only manner. Likewise,
    suffixing with ``_with_ghosts`` will grab all data on the rank (regular and
    ghost particles) in a read only array.

    All array-like properties return a `hoomd.array.HOOMDArray` object which
    prevents invalid memory accesses.
    """

    def __init__(self, state):
        super().__init__(state)
        self._particles = ParticleLocalAccessCPU(state)
        self._bonds = BondLocalAccessCPU(state)
        self._angles = AngleLocalAccessCPU(state)
        self._dihedrals = DihedralLocalAccessCPU(state)
        self._impropers = ImproperLocalAccessCPU(state)
        self._pairs = PairLocalAccessCPU(state)
        self._constraints = ConstraintLocalAccessCPU(state)
