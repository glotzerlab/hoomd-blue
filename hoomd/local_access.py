from abc import ABC, abstractmethod
from hoomd.hoomd_array import HOOMDArray
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

    _accessed_fields = dict()

    def __init__(self):
        object.__setattr__(self, '_entered', False)

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
        'type': 'getTypes',
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
        'tags': 'getTags',
        'rtags': 'getRTags'
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
        'constraint': 'getTypeVal',
        'group': 'getMembers',
        'tags': 'getTags',
        'rtags': 'getRTags'
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

    @property
    def particles(self):
        return self._particles

    @property
    def bonds(self):
        return self._bonds

    @property
    def angles(self):
        return self._angles

    @property
    def dihedrals(self):
        return self._dihedrals

    @property
    def impropers(self):
        return self._impropers

    @property
    def constraints(self):
        return self._constraints

    @property
    def pairs(self):
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
    def __init__(self, state):
        super().__init__(state)
        self._particles = ParticleLocalAccessCPU(state)
        self._bonds = BondLocalAccessCPU(state)
        self._angles = AngleLocalAccessCPU(state)
        self._dihedrals = DihedralLocalAccessCPU(state)
        self._impropers = ImproperLocalAccessCPU(state)
        self._pairs = PairLocalAccessCPU(state)
        self._constraints = ConstraintLocalAccessCPU(state)
