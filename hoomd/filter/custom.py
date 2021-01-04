from abc import ABCMeta, abstractmethod

class CustomFilter(metaclass=ABCMeta):
    """Abstract base class for custom particle filters.

    The class allows the definition of particle filters in Python (see
    `hoomd.filter.ParticleFilter`.
    """
    @abstractmethod
    def __call__(self, state):
        """Return the local particle tags that match the filter.

        This can either return the tags that are local to an MPI rank or the
        tags of all particles that match in the entire state. Tag numbers in a
        `hoomd.Snapshot` object are just their index. The full specification is
        that the set union of tags returned from each MPI rank will be used for
        the filter.
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
