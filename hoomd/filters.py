import hoomd._hoomd as _hoomd
import numpy as np


class ParticleFilter:
    def __hash__(self):
        return NotImplementedError("Must implement hash for Filters.")

    def __eq__(self, other):
        raise NotImplementedError(
            "Equality between {} is not defined.".format(self.__class__))

    def __str__(self):
        return "ParticleFilter.{}".format(self.__class__.__name__)

    def __call__(self, state):
        '''Needs to interact with state to get particles across MPI rank.'''
        raise NotImplementedError


class All(ParticleFilter, _hoomd.ParticleFilterAll):
    def __init__(self):
        _hoomd.ParticleFilterAll.__init__(self)

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return type(self) == type(other)


class Tags(ParticleFilter, _hoomd.ParticleFilterTags):
    def __init__(self, tags):
        if isinstance(tags, np.ndarray):
            tags = tags.astype(np.uint32)
        else:
            tags = np.array(tags, dtype=np.uint32)
        tags = np.unique(tags)
        self._tags = tags
        _hoomd.ParticleFilterTags.__init__(self, tags)

    def __hash__(self):
        if not hasattr(self, '_id'):
            self._id = hash(self._tags.tobytes())
        return self._id

    def __eq__(self, other):
        return all(self.tags == other.tags) and type(self) == type(other)

    @property
    def tags(self):
        return self._tags


class Types(ParticleFilter, _hoomd.ParticleFilterType):
    def __init__(self, types):
        types = set(types)
        self._types = frozenset(types)
        _hoomd.ParticleFilterType.__init__(self, types)

    def __hash__(self):
        return hash(self._types)

    def __eq__(self, other):
        return self._types == other._types and type(self) == type(other)

    @property
    def types(self):
        return self._types
