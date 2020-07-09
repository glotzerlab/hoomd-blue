from hoomd._hoomd import ParticleFilter


class _ParticleFilter(ParticleFilter):
    """Base class for all particle filters.

    Should not be instantiated or subclassed by users.
    """
    def __hash__(self):
        return NotImplementedError("Must implement hash for ParticleFilters.")

    def __eq__(self, other):
        raise NotImplementedError(
            "Equality between {} is not defined.".format(self.__class__))

    def __str__(self):
        return "ParticleFilter.{}".format(self.__class__.__name__)

    def __call__(self, state):
        """Needs to interact with state to get particles across MPI rank."""
        return self._get_selected_tags(state._cpp_sys_def)
