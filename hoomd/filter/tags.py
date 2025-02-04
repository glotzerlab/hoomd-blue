# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

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

    .. rubric:: Example:

    .. code-block:: python

        tags = hoomd.filter.Tags([0, 1, 2])
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
        return type(self) is type(other) and np.array_equal(
            self.tags, other.tags)

    @property
    def tags(self):
        """list[int]: List of particle tags to select."""
        return self._tags

    def __reduce__(self):
        """Enable (deep)copying and pickling of `Tags` particle filters."""
        return (type(self), (self.tags,))
