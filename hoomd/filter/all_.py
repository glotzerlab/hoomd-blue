# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define the All filter."""

from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterAll


class All(ParticleFilter, ParticleFilterAll):
    """Select all particles in the system.

    Base: `ParticleFilter`

    .. rubric:: Example:

    .. code-block:: python

        all_ = hoomd.filter.All()
    """

    def __init__(self):
        ParticleFilter.__init__(self)
        ParticleFilterAll.__init__(self)

    def __hash__(self):
        """Return a hash of the filter parameters."""
        return 0

    def __eq__(self, other):
        """Test for equality between two particle filters."""
        return type(self) is type(other)

    def __reduce__(self):
        """Enable (deep)copying and pickling of `All` particle filters."""
        return (type(self), tuple())
