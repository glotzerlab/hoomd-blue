# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define the Type filter."""

from hoomd.filter.filter_ import ParticleFilter
from hoomd._hoomd import ParticleFilterType


class Type(ParticleFilter, ParticleFilterType):
    """Select particles by type.

    Args:
        types (list[str]): List of particle type names to select.

    Base: `ParticleFilter`

    .. rubric:: Example:

    .. code-block:: python

        type_A_B = hoomd.filter.Type(['A', 'B'])
    """

    def __init__(self, types):
        ParticleFilter.__init__(self)
        types = set(types)
        self._types = frozenset(types)
        ParticleFilterType.__init__(self, types)

    def __hash__(self):
        """Return a hash of the filter parameters."""
        return hash(self._types)

    def __eq__(self, other):
        """Test for equality between two particle filters."""
        return type(self) is type(other) and self._types == other._types

    @property
    def types(self):
        """list[str]: List of particle type names to select."""
        return self._types

    def __reduce__(self):
        """Enable (deep)copying and pickling of `Type` particle filters."""
        return (type(self), (self.types,))
