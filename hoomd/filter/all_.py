"""Define the All filter."""

from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterAll


class All(ParticleFilter, ParticleFilterAll):
    """Select all particles in the system.

    Base: `ParticleFilter`
    """

    def __init__(self):
        ParticleFilter.__init__(self)
        ParticleFilterAll.__init__(self)

    def __hash__(self):
        """Return a hash of the filter parameters."""
        return 0

    def __eq__(self, other):
        """Test for equality between two particle filters."""
        return type(self) == type(other)

    def __reduce__(self):
        return (type(self), tuple())
