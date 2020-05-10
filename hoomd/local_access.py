from abc import ABC, abstractmethod
from hoomd._hoomd import LocalParticleData


class LocalAccess(ABC):
    @property
    @abstractmethod
    def _fields(self):
        pass

    _accessed_fields = dict()

    def __getattr__(self, attr):
        if attr in self._accessed_fields:
            return self._accessed_fields[attr]
        elif attr.startswith("ghost_"):
            if attr[6:] in self._fields:
                arr = getattr(self._cpp_obj, self._fields[attr[6:]])(
                    True, False)
            else:
                raise AttributeError(
                    "{} object has no attribute {}".format(type(self), attr))
        elif attr.endswith("_with_ghosts"):
            if attr[:-12] in self._fields:
                arr = getattr(self._cpp_obj, self._fields[attr[:-12]])(
                    False, True)
            else:
                raise AttributeError(
                    "{} object has no attribute {}".format(type(self), attr))
        elif attr in self._fields:
            arr = getattr(self._cpp_obj, self._fields[attr])(False, False)
        else:
            raise AttributeError(
                "{} object has no attribute {}".format(type(self), attr))
        self._accessed_fields[attr] = arr
        return arr

    def __setattr__(self, attr, value):
        raise RuntimeError(
            "Setting attributes directly to {} object is not allowed.".format(
                type(self)))

    def _enter(self):
        self._cpp_obj.enter()

    def _exit(self):
        self._cpp_obj.exit()


class ParticleLocalAccess(LocalAccess):
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
        object.__setattr__(
            self, '_cpp_obj',
            LocalParticleData(state._cpp_sys_def.getParticleData()))


class LocalSnapshot:
    def __init__(self, state):
        self._particles = ParticleLocalAccess(state)

    @property
    def particles(self):
        return self._particles

    def __enter__(self):
        self._particles._enter()
        return self

    def __exit__(self, type, value, traceback):
        self._particles._exit()
