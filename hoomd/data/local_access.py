# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Access simulation state data directly."""

from abc import ABC, abstractmethod
from hoomd import Box
from hoomd import _hoomd


class _LocalAccess(ABC):
    __slots__ = ('_entered', '_accessed_fields', '_cpp_obj')
    _global_fields = {'rtag': 'getRTags'}

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
                raise AttributeError("{} object has no attribute {}".format(
                    type(self), attr))

        self._accessed_fields[attr] = arr = self._array_cls(
            buff, lambda: self._entered)
        return arr

    def _get_raw_attr_and_flag(self, attr):
        ghosts_only = attr.startswith("ghost_")
        with_ghosts = attr.endswith("_with_ghost")
        raw_attr = attr.replace("_with_ghost", "").replace("ghost_", "")
        if ghosts_only and with_ghosts:
            raise ValueError("Attribute cannot be both prefixed with ghost_ "
                             "and suffixed with _with_ghost")
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
            raise AttributeError("{} object has no attribute {}.".format(
                self.__class__, attr))
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
    """Class for directly accessing HOOMD-blue particle data.

    Attributes:
        typeid ((N_particles) `hoomd.data.array` object of ``float``):
            The integer type of a particle
        tag ((N_particles) `hoomd.data.array` object of ``int``):
            The tag of a particle.  HOOMD-blue uses spacial sorting to improve
            cache efficiency in particle look-ups. This means the ordering of
            the array changes.  However, particle tags remain constant. This
            means that if ``particles.tag[0]`` is 1, then later whatever
            particle has a tag of 1 later in the simulation is the same
            particle.
        rtag ((N_particles_global) `hoomd.data.array` object of ``int``):
            The reverse tag of a particle. This means that the value
            ``particles.rtag[0]`` represents the current index accessing data
            for the particle with tag 0.
        position ((N_particles, 3) `hoomd.data.array` object of ``float``):
            particle positions :math:`[\\mathrm{length}]`
        image ((N_particles, 3) `hoomd.data.array` object of ``int``):
            The periodic image a particle occupies
        velocity ((N_particles, 3) `hoomd.data.array` object of ``float``):
            particle velocities :math:`[\\mathrm{velocity}]`
        acceleration ((N_particles, 3) `hoomd.data.array` object of ``float``):
            particle accelerations
            :math:`[\\mathrm{velocity} \\cdot \\mathrm{time}^{-1}]`
        mass ((N_particles) `hoomd.data.array` object of ``float``):
            particles' masses :math:`[\\mathrm{mass}]`
        orientation ((N_particles, 4) `hoomd.data.array` object of ``float``):
            particle orientations expressed as quaternions
        angmom ((N_particles, 4) `hoomd.data.array` object of \
            ``float``):
            particle angular momenta expressed as quaternions
            :math:`[\\mathrm{mass} \\cdot \\mathrm{velocity} \\cdot
            \\mathrm{length}]`
        moment_inertia ((N_particles, 3) `hoomd.data.array` object of \
            ``float``):
            particle principal moments of inertia
            :math:`[\\mathrm{mass} \\cdot \\mathrm{length}^2]`
        charge ((N_particles) `hoomd.data.array` object of ``float``):
            particle electrical charges :math:`[\\mathrm{charge}]`
        diameter ((N_particles) `hoomd.data.array` object of ``float``):
            particle diameters :math:`[\\mathrm{length}]`
        body ((N_particles) `hoomd.data.array` object of ``int``):
            The id of the rigid body the particle is in.
        net_force ((N_particles, 3) `hoomd.data.array` object of ``float``):
            net force on particle :math:`[\\mathrm{force}]`
        net_torque ((N_particles, 3) `hoomd.data.array` object of ``float``):
            net torque on particle
            :math:`[\\mathrm{force} \\cdot \\mathrm{length}]`
        net_virial ((N_particles, 3) `hoomd.data.array` object of ``float``):
            net virial on particle :math:`[\\mathrm{energy}]`
        net_energy ((N_particles,) `hoomd.data.array` object of ``float``):
            net energy of a particle (accounts for duplicate counting of an
            interaction). :math:`[\\mathrm{energy}]`

    Note:
        That changing some attributes like (``velocity`` or ``acceleration``)
        may not change the trajectory of the system. Integration of the
        equations of motion do not necessarily use velocity or accleration
        directly. This is also true in HOOMD-blue's MD integration methods (see
        `hoomd.md.methods`)
    """

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
        'angmom': 'getAngularMomentum',
        'moment_inertia': 'getMomentsOfInertia',
        'charge': 'getCharge',
        'diameter': 'getDiameter',
        'image': 'getImages',
        'tag': 'getTags',
        'rtag': 'getRTags',
        'body': 'getBodies',
        'net_force': 'getNetForce',
        'net_torque': 'getNetTorque',
        'net_virial': 'getNetVirial',
        'net_energy': 'getNetEnergy'
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
        'typeid': 'getTypeVal',
        'group': 'getMembers',
        'tag': 'getTags',
        'rtag': 'getRTags'
    }

    def __init__(self, state):
        super().__init__()
        self._cpp_obj = self._cpp_cls(
            getattr(state._cpp_sys_def, self._cpp_get_data_method_name)())


class BondLocalAccessBase(_GroupLocalAccess):
    """Class for directly accessing HOOMD-blue bond data.

    Attributes:
        typeid ((N_bonds) `hoomd.data.array` object of ``int``):
            The integer type of a bond.
        members ((N_bonds, 2) `hoomd.data.array` object of ``int``):
            The tags of particles in a bond.
        tag ((N_bonds) `hoomd.data.array` object of ``int``):
            The tag of the bond.  HOOMD-blue uses spacial sorting to improve
            cache efficiency in bond look-ups. This means the ordering of the
            array changes.  However, bond tags remain constant. This means that
            if ``bond.tag[0]`` is 1, then later whatever bond has a tag of 1
            later in the simulation is the same bond.
        rtag ((N_bonds_global) `hoomd.data.array` object of ``int``): the
            reverse tag of a bond. This means that the value ``bond.rtag[0]``
            represents the current index to access data for the bond with tag 0.
    """
    _cpp_get_data_method_name = "getBondData"


class AngleLocalAccessBase(_GroupLocalAccess):
    """Class for directly accessing HOOMD-blue angle data.

    Attributes:
        typeid ((N_angles) `hoomd.data.array` object of ``int``):
            The integer type of a angle.
        members ((N_angles, 3) `hoomd.data.array` object of ``int``):
            The tags of particles in a angle.
        tag ((N_angles) `hoomd.data.array` object of ``int``):
            The tag of the angle.  HOOMD-blue uses spacial sorting to improve
            cache efficiency in angle look-ups. This means the ordering of the
            array changes.  However, angle tags remain constant. This means
            that if ``angle.tag[0]`` is 1, then later whatever angle has a
            tag of 1 later in the simulation is the same angle.
        rtag ((N_angles_global) `hoomd.data.array` object of ``int``):
            The reverse tag of a angle. This means that the value
            ``angle.rtag[0]`` represents the current index for accessing data
            for the angle with tag 0.
    """
    _cpp_get_data_method_name = "getAngleData"


class DihedralLocalAccessBase(_GroupLocalAccess):
    """Class for directly accessing HOOMD-blue dihedral data.

    Attributes:
        typeid ((N_dihedrals) `hoomd.data.array` object of ``int``): The integer
            type of a dihedral.
        members ((N_dihedrals, 3) `hoomd.data.array` object of ``int``): the
            tags of particles in a dihedral.
        tag ((N_dihedrals) `hoomd.data.array` object of ``int``):
            The tag of the dihedral.  HOOMD-blue uses spacial sorting to improve
            cache efficiency in dihedral look-ups. This means the ordering of
            the array changes.  However, dihedral tags remain constant. This
            means that if ``dihedral.tag[0]`` is 1, then later whatever dihedral
            has a tag of 1 later in the simulation is the same dihedral.
        rtag ((N_dihedrals_global) `hoomd.data.array` object of ``int``):
            The reverse tag of a dihedral. This means that the value
            ``dihedral.rtag[0]`` represents the current index for accessing data
            for the dihedral with tag 0.
    """
    _cpp_get_data_method_name = "getDihedralData"


class ImproperLocalAccessBase(_GroupLocalAccess):
    """Class for directly accessing HOOMD-blue improper data.

    Attributes:
        typeid ((N_impropers) `hoomd.data.array` object of ``int``):
            The integer type of a improper.
        members ((N_impropers, 3) `hoomd.data.array` object of ``int``):
            The tags of particles in a improper.
        tag ((N_impropers) `hoomd.data.array` object of ``int``):
            The tag of the improper.  HOOMD-blue uses spacial sorting to improve
            cache efficiency in improper look-ups. This means the ordering of
            the array changes.  However, improper tags remain constant. This
            means that if ``improper.tag[0]`` is 1, then later whatever improper
            has a tag of 1 later in the simulation is the same improper.
        rtag ((N_impropers_global) `hoomd.data.array` object of ``int``):
            The reverse tag of a improper. This means that the value
            ``improper.rtag[0]`` represents the current index for accessing data
            for the improper with tag 0.
    """
    _cpp_get_data_method_name = "getImproperData"


class ConstraintLocalAccessBase(_GroupLocalAccess):
    """Class for directly accessing HOOMD-blue constraint data.

    Attributes:
        value ((N_constraints) `hoomd.data.array` object of ``float``): The
            constaint value.
        members ((N_constraints, 3) `hoomd.data.array` object of ``int``): the
            tags of particles in a constraint.
        tag ((N_constraints) `hoomd.data.array` object of ``int``):
            The tag of the constraint.  HOOMD-blue uses spacial sorting to
            improve cache efficiency in constraint look-ups. This means the
            ordering of the array changes.  However, constraint tags remain
            constant. This means that if ``constraint.tag[0]`` is 1, then later
            whatever constraint has a tag of 1 later in the simulation is the
            same constraint.
        rtag ((N_constraints_global) `hoomd.data.array` object of ``int``):
            The reverse tag of a constraint. This means that the value
            ``constraint.rtag[0]`` represents the current index for accessing
            data for the constraint with tag 0.
    """
    _fields = {
        'value': 'getTypeVal',
        'group': 'getMembers',
        'tag': 'getTags',
        'rtag': 'getRTags'
    }
    _cpp_get_data_method_name = "getConstraintData"


class PairLocalAccessBase(_GroupLocalAccess):
    """Class for directly accessing HOOMD-blue special pair data.

    Attributes:
        typeid ((N_pairs) `hoomd.data.array` object of ``float``): The type of
            special pair.
        members ((N_pairs, 3) `hoomd.data.array` object of ``int``): the tags of
            particles in a special pair.
        tag ((N_special_pairs) `hoomd.data.array` object of ``int``):
            The tag of the special pair.  HOOMD-blue uses spacial sorting to
            improve cache efficiency in special pair look-ups. This means the
            ordering of the array changes.  However, special pair tags remain
            constant. This means that if ``special pair.tag[0]`` is 1, then
            later whatever special pair has a tag of 1 later in the simulation
            is the same special pair.
        rtag ((N_special_pairs_global) `hoomd.data.array` object of ``int``):
            The reverse tag of a special pair. This means that the value
            ``special pair.rtag[0]`` represents the current index for accessing
            data for the special pair with tag 0.
    """
    _cpp_get_data_method_name = "getPairData"


class _LocalSnapshot:

    def __init__(self, state, call_update_group_dof):
        self._state = state
        self._box = state.box
        self._local_box = state._cpp_sys_def.getParticleData().getBox()
        self._call_update_group_dof = call_update_group_dof

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
        if self._call_update_group_dof:
            self._state.update_group_dof
