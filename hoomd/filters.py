import hoomd._hoomd as _hoomd
import numpy as np


class ParticleFilter:
    def __init__(self, *args, **kwargs):
        self._id = self._get_id(*args, **kwargs)

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        raise NotImplementedError(
            "Equality between {} is not defined.".format(self.__class__))

    def __str__(self):
        return "ParticleFilter.{}".format(self.__class__.__name__)

    def __call__(self, state):
        '''Needs to interact with state to get particles across MPI rank.'''
        raise NotImplementedError

    def _compute_id(self, *args, **kwargs):
        raise NotImplementedError(
            "_compute_id for {} is not defined.".format(self.__class__))

    @property
    def id(self):
        return self._id


class All(ParticleFilter, _hoomd.ParticleFilterAll):
    def __init__(self):
        ParticleFilter.__init__(self)
        _hoomd.ParticleFilterAll.__init__(self)

    def _compute_id(self):
        return 0

    def __eq__(self, other):
        return type(self) == type(other)


class Tags(ParticleFilter, _hoomd.ParticleFilterTags):
    def __init__(self, tags):
        if isinstance(tags, np.ndarray):
            tags = tags.astype(np.uint32)
        ParticleFilter.__init__(self, tags)
        _hoomd.ParticleFilterTags.__init__(self, tags)

    def _compute_id(self, tags):
        return hash(tags.tobytes())

    def __eq__(self, other):
        return self.tags == other.tags and type(self) == type(other)
