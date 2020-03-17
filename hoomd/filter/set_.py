from hoomd.filter.filter_ import ParticleFilter
from hoomd import _hoomd


class ParticleFilterSetOperations(ParticleFilter):
    def __init__(self, f, g):
        if f == g:
            raise ValueError("Cannot use same filter for {}"
                             "".format(self.__class__.__name__))
        else:
            self._f = f
            self._g = g
        getattr(_hoomd, self._cpp_cls_name).__init__(self, f, g)

    def __hash__(self):
        return hash(hash(self._f) + hash(self._g))

    def __eq__(self, other):
        return type(self) == type(other) and \
            self._f == other._f and \
            self._g == other._g


class SetDifference(ParticleFilterSetOperations,
                    _hoomd.ParticleFilterSetDifference):
    _cpp_cls_name = 'ParticleFilterSetDifference'


class Union(ParticleFilterSetOperations, _hoomd.ParticleFilterUnion):
    _cpp_cls_name = 'ParticleFilterUnion'


class Intersection(ParticleFilterSetOperations,
                   _hoomd.ParticleFilterIntersection):
    _cpp_cls_name = 'ParticleFilterIntersection'
