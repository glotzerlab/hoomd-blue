"""Define the Tags filter."""

from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterTags
import numpy as np


class Tags(ParticleFilter, ParticleFilterTags):
    """Select particles by tag.

    Args:
        tags (list[int]): List of particle tags to select.

    Base: `ParticleFilter`
    """

    def __init__(self, tags):
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
