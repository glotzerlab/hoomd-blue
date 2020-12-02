"""Define the Tags filter."""

from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterTags
import numpy as np


class Tags(ParticleFilter, ParticleFilterTags):
    """Select particles by tag.

    Args:
        tags (list[int]): List of particle tags to select.

    A particle tag is a unique identifier assigned to each particle in the
    simulation state. When the state is first initialized, it assigns tags
    0 through `N_particles` to the particles in the order provided.

    Base: `ParticleFilter`
    """

    def __init__(self, tags):
        ParticleFilter.__init__(self)
        self._tags = np.ascontiguousarray(np.unique(tags), dtype=np.uint32)
        ParticleFilterTags.__init__(self, tags)

    def __hash__(self):
        """Return a hash of the filter parameters."""
        if not hasattr(self, '_id'):
            self._id = hash(self._tags.tobytes())
        return self._id

    def __eq__(self, other):
        """Test for equality between two particle filters."""
        return type(self) == type(other) and all(self.tags == other.tags)

    @property
    def tags(self):
        """list[int]: List of particle tags to select."""
        return self._tags

    def __reduce__(self):
        return (type(self), (self.tags,))
