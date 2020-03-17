from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterAll


class All(ParticleFilter, ParticleFilterAll):
    def __init__(self):
        ParticleFilterAll.__init__(self)

    def __hash__(self):
        return 0

    def __eq__(self, other):
        return type(self) == type(other)
