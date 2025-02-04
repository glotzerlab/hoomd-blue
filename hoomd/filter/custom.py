# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Contains a class for custom particle filters in Python.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
"""
from abc import abstractmethod
from collections.abc import Hashable, Callable


class CustomFilter(Hashable, Callable):
    """Base class for custom particle filters.

    The class allows the definition of particle filters in Python (see
    `ParticleFilter`).

    Subclasses of this class must implement ``__hash__``, ``__eq__``, and
    ``__call__`` methods. The ``__hash__`` and ``__eq__`` methods will be used
    to cache the particle tags associated with a filter, thus ``__eq__`` must
    correctly disambiguate any filters that would choose different particles.
    For more information on the Python data model see
    `<https://docs.python.org/3/reference/datamodel.html#object.__hash__>`_ and
    `<https://docs.python.org/3/reference/datamodel.html#object.__eq__>`_.

    .. rubric:: Example:

    .. code-block:: python

        class MassRangeFilter(hoomd.filter.CustomFilter):
            def __init__(self, min_mass, max_mass):
                self.min_mass = min_mass
                self.max_mass = max_mass

            def __hash__(self):
                return hash((self.min_mass, self.max_mass))

            def __eq__(self, other):
                return (isinstance(other, MassRangeFilter)
                        and self.min_mass == other.min_mass
                        and self.max_mass == other.max_mass)

            def __call__(self, state):
                with state.cpu_local_snapshot as snap:
                    masses = snap.particles.mass
                    indices = ((masses > self.min_mass)
                               & (masses < self.max_mass))
                    return numpy.copy(snap.particles.tag[indices])

        mass_range_filter = MassRangeFilter(1.0, 5.0)
        print(mass_range_filter(simulation.state))

    Warning:
        Custom filters will not work with the set operation particle filters
        (i.e.  `hoomd.filter.Union`, `hoomd.filter.Intersection`, or
        `hoomd.filter.SetDifference`). Implement any needed logic in your
        `__call__` method.
    """

    @abstractmethod
    def __call__(self, state):
        """Return the local particle tags that match the filter.

        Returns the tags that match the particle filter. Tag numbers in a
        `hoomd.Snapshot` object are equal to their index.

        Note:
            In MPI domain decomposition simulation, the set union of the
            returned arrays from each MPI rank must contain all particle tags
            that match the filter.


        Tip:
            To obtain the best performance, use local snaphots to access
            particle data, locally evaluate the filter on each rank, and return
            the list of local tags that match. `__call__` should not perform the
            set union, that is the caller's responsibility.

        Args:
            state (`hoomd.State`):
                The simulation state to return the filtered tags from.

        Returns:
            (N,) `numpy.ndarray` of `numpy.uint64`:
                An array of particle tags filter.
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """Whether this filter and another filter are equal.

        This is necessary to allow for proper caching of filter tags in
        `hoomd.State`.
        """
        pass
