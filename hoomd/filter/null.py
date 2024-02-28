# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define the Null filter."""

from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterNull


class Null(ParticleFilter, ParticleFilterNull):
    """Select no particles.

    Base: `ParticleFilter`

    .. rubric:: Example:

    .. code-block:: python

        null = hoomd.filter.Null()
    """

    def __init__(self):
        ParticleFilter.__init__(self)
        ParticleFilterNull.__init__(self)

    def __hash__(self):
        """Return a hash of the filter parameters."""
        return 0

    def __eq__(self, other):
        """Test for equality between two particle filters."""
        return type(self) is type(other)
