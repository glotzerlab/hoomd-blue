# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define ParticleFilter."""

from hoomd import _hoomd


class ParticleFilter(_hoomd.ParticleFilter):
    """Base class for all particle filters.

    This class provides methods common to all particle filters.

    Attention:
        Users should instantiate one of the subclasses. Calling `ParticleFilter`
        directly may result in an error.
    """

    def __hash__(self):
        """Return a hash of the filter parameters."""
        return NotImplementedError("Must implement hash for ParticleFilters.")

    def __eq__(self, other):
        """Test for equality between two particle filters."""
        raise NotImplementedError("Equality between {} is not defined.".format(
            self.__class__))

    def __str__(self):
        """Format a human readable string describing the filter."""
        return "ParticleFilter.{}".format(self.__class__.__name__)

    def __call__(self, state):
        """Evaluate the filter.

        Returns:
            list[int]: The particle tags selected by this filter.

        Note:
            This method may return tags that are only present on the local MPI
            rank. The full set of particles selected is the combination of
            these the lists across ranks with a set union operation.
        """
        return self._get_selected_tags(state._cpp_sys_def)
