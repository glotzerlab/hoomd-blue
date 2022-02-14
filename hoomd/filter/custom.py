# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Contains a class for custom particle filters in Python."""
from abc import abstractmethod
from collections.abc import Hashable, Callable


class CustomFilter(Hashable, Callable):
    """Abstract base class for custom particle filters.

    The class allows the definition of particle filters in Python (see
    `ParticleFilter`).

    Subclasses of this class must have ``__hash__``, ``__eq__``, and
    ``__call__`` methods. The ``__hash__`` and ``__eq__`` methods will be used
    to cache the particle tags associated with a filter, thus ``__eq__`` must
    correctly disambiguity any filters that would choose different particles.
    For more information on the Python data model see
    `<https://docs.python.org/3/reference/datamodel.html#object.__hash__>`_ and
    `<https://docs.python.org/3/reference/datamodel.html#object.__eq__>`_.

    The example below creates a custom filter that filters out particles above
    and below a certain mass, and uses that filter in a `hoomd.write.GSD`
    object.

    Example::

        class MassFilter(hoomd.filter.CustomFilter):
            def __init__(self, min_mass, max_mass):
                self.min_mass = min_mass
                self.max_mass = max_mass

            def __hash__(self):
                return hash((self.min_mass, self.max_mass))

            def __eq__(self, other):
                return (isinstance(other, DiameterFilter)
                        and self.min_mass == other.min_mass
                        and self.max_mass == other.max_mass)

            def __call__(self, state):
                with state.cpu_local_snapshot as snap:
                    masses = snap.particles.mass
                    indices = ((masses > self.min_mass)
                               & (masses < self.max_mass))
                    return np.copy(snap.particles.tag[indices])

        # All particles with 1.0 < mass < 5.0
        filter_ = DiameterFilter(1.0, 5.0)
        gsd = hoomd.write.GSD('example.gsd', 100, filter=filter_)

    Warning:
        Custom filters will not work with the set operation particle filters
        (i.e.  `hoomd.filter.Union`, `hoomd.filter.Intersection`, or
        `hoomd.filter.SetDifference`). This restriction may be lifted in a
        future version.
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
    def __eq__(self, other):
        """Whether this filter and another filter are equal.

        This is necessary to allow for proper caching of filter tags internally
        in HOOMD-blue.
        """
        pass
