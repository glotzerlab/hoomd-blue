from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterType


class Type(ParticleFilter, ParticleFilterType):
    def __init__(self, types):
        types = set(types)
        self._types = frozenset(types)
        ParticleFilterType.__init__(self, types)

    def __hash__(self):
        return hash(self._types)

    def __eq__(self, other):
        return type(self) == type(other) and self._types == other._types

    @property
    def types(self):
        return self._types
