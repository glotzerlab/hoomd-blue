import hoomd._hoomd as _hoomd
import numpy as np


class ParticleFilter:

    def __init__(self, *args, **kwargs):
        args_str = ''.join([repr(arg) if not isinstance(arg, np.ndarray)
                            else arg.tostring() for arg in args])
        kwargs_str = ''.join([repr(value) if not isinstance(value, np.ndarray)
                             else value.tostring()
                             for value in kwargs.values()])
        self.args_str = args_str
        self.kwargs_str = kwargs_str
        _id = hash(self.__class__.__name__ + args_str + kwargs_str)
        self._id = _id

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self._id == other._id

    def __str__(self):
        return "ParticleFilter.{}".format(self.__class__.__name__)

    def __call__(self, state):
        '''Needs to interact with state to get particles across MPI rank.'''
        raise NotImplementedError


class All(ParticleFilter, _hoomd.ParticleFilterAll):
    def __init__(self):
        ParticleFilter.__init__(self)
        _hoomd.ParticleFilterAll.__init__(self)


class Tags(ParticleFilter, _hoomd.ParticleFilterTags):
    def __init__(self, tags):
        if isinstance(tags, np.ndarray):
            tags = tags.astype(np.uint32)
        ParticleFilter.__init__(self, tags)
        _hoomd.ParticleFilterTags.__init__(self, tags)
