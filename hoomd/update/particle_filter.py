# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement Python interface for a ParticleFilter updater.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    filter1 = hoomd.filter.All()
    filter2 = hoomd.filter.All()
"""

import copy

import hoomd


class _GroupConverter:
    """Used to go from filter to group for synced list Python <-> C++ syncing.

    We have to use a callable with state because the callable needs to know how
    to get the ParticleGroup associated with a filter. As the groups are not a
    user facing object, we need to use the internal group storage of the state
    object which is available until adding/attaching.
    """

    def __call__(self, filter):
        return self._state._get_group(filter)

    def _attach(self, simulation):
        self._state = simulation.state

    def _detach(self):
        self._state = None

    def __getstate__(self):
        """Necessary to pickle FilterUpdater."""
        state = copy.copy(self.__dict__)
        state.pop("_state", None)
        return state


class FilterUpdater(hoomd.operation.Updater):
    """Update sets of particles associated with a filter.

    Args:
        trigger (hoomd.trigger.trigger_like): A trigger to use for determining
            when to update particles associated with a filter.
        filters (list[hoomd.filter.filter_like]): A list of
            `hoomd.filter.filter_like` objects to update.

    `hoomd.Simulation` caches the particles selected by
    `hoomd.filter.filter_like` objects to avoid the cost of re-running the
    filter on every time step. The particles selected by a filter will remain
    static until recomputed. This class provides a mechanism to update the
    cached list of particles. For example, periodically update a MD integration
    method's group so that the integration method applies to particles in a
    given region of space.

    Tip:
        To improve performance, use a `hoomd.trigger.Trigger` subclass, to
        update only when there is a known change to the particles that a filter
        would select.

    Note:
        Some actions automatically recompute all filter particles such as adding
        or removing particles to the `hoomd.Simulation.state`.

    .. rubric:: Example:

    .. code-block:: python

        filter_updater = hoomd.update.FilterUpdater(
            trigger=hoomd.trigger.Periodic(1_000),
            filters=[filter1, filter2])
    """

    def __init__(self, trigger, filters):
        super().__init__(trigger)
        self._filters = hoomd.data.syncedlist.SyncedList(
            hoomd.filter.ParticleFilter,
            iterable=filters,
            to_synced_list=_GroupConverter(),
            attach_members=False)

    @property
    def filters(self):
        """list[hoomd.filter.filter_like]: filters to update select \
                particles.

        .. rubric:: Example:

        .. code-block:: python

            filter_updater.filters = [filter1, filter2]
        """
        return self._filters

    @filters.setter
    def filters(self, new_filters):
        self._filters.clear()
        self._filters.extend(new_filters)

    def _attach_hook(self):
        # We use a SyncedList internal API to allow for storing the state to
        # query groups from filters.
        self._filters._to_synced_list_conversion._attach(self._simulation)
        self._cpp_obj = hoomd._hoomd.ParticleFilterUpdater(
            self._simulation.state._cpp_sys_def, self.trigger)
        self._filters._sync(self._simulation, self._cpp_obj.groups)

    def __eq__(self, other):
        """Return whether two objects are equivalent."""
        return super().__eq__(other) and self._filters == other._filters
