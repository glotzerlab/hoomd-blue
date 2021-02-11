"""Contains a class for custom particle filters in Python."""
from abc import abstractmethod
from collections.abc import Hashable, Callable

class CustomFilter(Hashable, Callable):
    """Abstract base class for custom particle filters.

    The class allows the definition of particle filters in Python (see
    `hoomd.filter.ParticleFilter`).
    """
    @abstractmethod
    def __call__(self, state):
        """Return the local particle tags that match the filter.

        Returns the tags that are local to an MPI rank that match the particle
        filter. Tag numbers in a `hoomd.Snapshot` object are just their index.

        Note:
            The exact requirements for the tags returned by custom filters on
            each MPI rank is that the set union of the returned arrays from each
            MPI rank be all particles that match the filter. For general use,
            however, it is recommended that each rank only return the tags for
            particles that are in the local MPI rank (excluding ghost
            particles). This is preferable for ease of use with local snapshots
            to avoid accidentally attempting to access invalid array indices
            from tags outside of the MPI rank.

        Args:
            state (`hoomd.State`):
                The simulation state to return the filtered tags from.

        Returns:
            (N,) `numpy.ndarray` of `numpy.uint64`:
                An array of MPI local tags that match the filter.
        """
        pass

    @abstractmethod
    def __hash__(self):
        """A hashed value to represent this instance.

        This is necessary to allow for proper caching of filter tags internally
        in HOOMD-blue.
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """Whether this filter and another filter are equal.

        This is necessary to allow for proper caching of filter tags internally
        in HOOMD-blue.
        """
        pass
