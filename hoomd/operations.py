# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement a storage and management class for HOOMD-blue operations.

Defines the `Operations` class which serves as the main class for storing and
organizing the many parts of a simulation in a way that allows operations to be
added and removed from a `Simulation`.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    operation = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1000),
                                filename=tmp_path / 'operations.gsd',
                                filter=hoomd.filter.All())
"""

# Operations also automatically handles attaching and detaching (creating and
# destroying C++ objects) for all hoomd operations.

import weakref
from collections.abc import Collection
from copy import copy
from itertools import chain
from hoomd.data import syncedlist
from hoomd.operation import Writer, Updater, Tuner, Compute, Integrator
from hoomd.tune import ParticleSorter
from hoomd.error import DataAccessError
from hoomd import _hoomd


class Operations(Collection):
    """A mutable collection of operations which act on a `Simulation`.

    An `Operations` class instance contains all the operations acting on a
    simulation. These operations are classes that perform various actions on a
    `hoomd.Simulation`. Operations can be added and removed at any point from a
    `hoomd.Operations` instance. The class provides the interface defined by
    `collections.abc.Collection`. Other methods for manipulating instances mimic
    Python objects where possible, but the class is not simply a mutable list or
    set. `Operations` objects manage multiple independent sequences described
    below.

    The types of operations which can be added to an `Operations` object are
    tuners, updaters, integrators, writers, and computes. An `Operations`
    instance can have zero or one integrator and any number of tuners, updaters,
    writers, or computes. To see examples of these types of operations see
    `hoomd.tune` (tuners), `hoomd.update` (updaters), `hoomd.hpmc.integrate` or
    `hoomd.md.Integrator` (integrators), `hoomd.write` (writers), and
    `hoomd.md.compute.ThermodynamicQuantities` (computes).

    A given instance of an operation class can only be added to a single
    `Operations` container. Likewise, a single instance cannot be added to the
    same `Operations` container more than once.

    All `Operations` instances start with a `hoomd.tune.ParticleSorter` instance
    in their ``tuners`` attribute. This increases simulation
    performance. However, users can choose to modify or remove this tuner if
    desired.

    Note:
        An `Operations` object is created by default when a new simulation is
        created.
    """

    def __init__(self):
        self._scheduled = False
        self._simulation = None
        sync_func = syncedlist._PartialGetAttr('_cpp_obj')
        self._updaters = syncedlist.SyncedList(Updater, sync_func)
        self._writers = syncedlist.SyncedList(Writer, sync_func)
        self._tuners = syncedlist.SyncedList(Tuner, sync_func)
        self._computes = syncedlist.SyncedList(Compute, sync_func)
        self._integrator = None
        self._tuners.append(ParticleSorter())

    def _get_proper_container(self, operation):
        if isinstance(operation, Updater):
            return self._updaters
        elif isinstance(operation, Writer):
            return self._writers
        elif isinstance(operation, Tuner):
            return self._tuners
        elif isinstance(operation, Compute):
            return self._computes
        else:
            raise TypeError(f"{type(operation)} is not a valid operation type.")

    def add(self, operation):
        """Add an operation to this container.

        Adds the provided operation to the appropriate attribute of the
        `Operations` instance.

        Args:
            operation (hoomd.operation.Operation): A HOOMD-blue tuner,
                updater, integrator, writer, or compute to add to the
                collection.

        Raises:
            TypeError: If ``operation`` is not of a valid type.

        Note:
            Since only one integrator can be associated with an `Operations`
            object at a time, this removes the current integrator when called
            with an integrator operation. Also, the ``integrator`` property
            cannot be set to ``None`` using this function. Use
            ``operations.integrator = None`` explicitly for this.

        .. rubric:: Example:

        .. code-block:: python

            simulation.operations.add(operation)
        """
        # we raise this error here to provide a more clear error message.
        if isinstance(operation, Integrator):
            self.integrator = operation
        else:
            try:
                container = self._get_proper_container(operation)
            except TypeError:
                raise TypeError(f"Type {type(operation)} is not a valid "
                                f"type to add to Operations.")
            container.append(operation)

    def __iadd__(self, operation):
        """Works the same as `Operations.add`.

        Args:
            operation (hoomd.operation.Operation): A HOOMD-blue tuner,
                updater, integrator, writer, or compute to add to the object.

        .. rubric:: Example:

        .. code-block:: python

            simulation.operations += operation
        """
        self.add(operation)
        return self

    def remove(self, operation):
        """Remove an operation from the `Operations` object.

        Remove the item from the collection whose Python object `id` is the same
        as ``operation``.

        Args:
            operation (hoomd.operation.Operation): A HOOMD-blue integrator,
                tuner, updater, integrator, or compute to remove from the
                container.

        Raises:
            ValueError: If ``operation`` is not found in this container.
            TypeError: If ``operation`` is not of a valid type.

        .. rubric:: Example:

        .. code-block:: python

            simulation.operations.remove(operation)
        """
        if isinstance(operation, Integrator):
            self.integrator = None
        else:
            try:
                container = self._get_proper_container(operation)
            except TypeError:
                raise TypeError(f"Type {type(operation)} is not a valid "
                                f"type to remove from Operations.")
            container.remove(operation)

    def __isub__(self, operation):
        """Works the same as `Operations.remove`.

        Args:
            operation (hoomd.operation.Operation): A HOOMD-blue integrator,
                tuner, updater, integrator, analyzer, or compute to remove from
                the collection.

        .. rubric:: Example:

        .. code-block:: python

            simulation.operations -= operation
        """
        self.remove(operation)
        return self

    @property
    def _sys_init(self):
        if self._simulation is None or self._simulation.state is None:
            return False
        else:
            return True

    def _schedule(self):
        """Prepares all operations for a `hoomd.Simulation.run` call.

        Creates the internal C++ objects for all operations.

        Raises:
            RuntimeError: raises when not associated with a `hoomd.Simulation`
                object.
        """
        if not self._sys_init:
            raise RuntimeError("System not initialized yet")
        sim = self._simulation
        if not (self.integrator is None or self.integrator._attached):
            self.integrator._attach(sim)
        if not self.updaters._synced:
            self.updaters._sync(sim, sim._cpp_sys.updaters)
        if not self.tuners._synced:
            self.tuners._sync(sim, sim._cpp_sys.tuners)
        if not self.computes._synced:
            self.computes._sync(sim, sim._cpp_sys.computes)
        if not self.writers._synced:
            self.writers._sync(sim, sim._cpp_sys.analyzers)
        self._scheduled = True

    def _unschedule(self):
        """Undo the effects of `Operations._schedule`."""
        if self.integrator is not None:
            self._integrator._detach()
        self._writers._unsync()
        self._updaters._unsync()
        self._tuners._unsync()
        self._computes._unsync()
        self._scheduled = False

    def __contains__(self, operation):
        """Whether an operation is contained in this container.

        Args:
            operation: Returns whether this exact operation is
                contained in the collection.

        .. rubric:: Example:

        .. code-block:: python

            operation in simulation.operations
        """
        return any(op is operation for op in self)

    def __iter__(self):
        """Iterates through all contained operations.

        .. rubric:: Example:

        .. code-block:: python

            for operation in simulation.operations:
                pass
        """
        integrator = (self._integrator,) if self._integrator else []
        yield from chain(self._tuners, self._updaters, integrator,
                         self._writers, self._computes)

    def __len__(self):
        """Return the number of operations contained in this collection.

        .. rubric:: Example:

        .. code-block:: python

            len(simulation.operations)
        """
        base_len = len(self._writers) + len(self._updaters) + len(self._tuners)
        return base_len + (1 if self._integrator is not None else 0)

    @property
    def integrator(self):
        """`hoomd.operation.Integrator`: An MD or HPMC integrator object.

        `Operations` objects have an initial ``integrator`` property of
        ``None``. Can be set to MD or HPMC integrators. The property can also be
        set to ``None``.

        .. rubric:: Examples:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            simulation.operations.integrator = hoomd.md.Integrator(dt=0.001)

        .. code-block:: python

            simulation.operations.integrator = None
        """
        return self._integrator

    @integrator.setter
    def integrator(self, op):
        if op is not None:
            if not isinstance(op, Integrator):
                raise TypeError("Cannot set integrator to a type not derived "
                                "from hoomd.operation.Integrator")
        old_ref = self.integrator
        self._integrator = op
        # Handle attaching and detaching integrators dealing with None values
        if self._scheduled:
            if op is not None:
                op._attach(self._simulation)
            if old_ref is not None:
                old_ref._detach()

    @property
    def updaters(self):
        """list[`hoomd.operation.Updater`]: A list of updater operations.

        Holds the list of updaters associated with this collection. The list can
        be modified as a standard Python list.
        """
        return self._updaters

    @property
    def writers(self):
        """list[`hoomd.operation.Writer`]: A list of writer operations.

        Holds the list of writers associated with this collection. The list
        can be modified as a standard Python list.
        """
        return self._writers

    @property
    def tuners(self):
        """list[`hoomd.operation.Tuner`]: A list of tuner operations.

        Holds the list of tuners associated with this collection. The list can
        be modified as a standard Python list.
        """
        return self._tuners

    @property
    def computes(self):
        """list[`hoomd.operation.Compute`]: A list of compute operations.

        Holds the list of computes associated with this collection. The list
        can be modified as a standard Python list.
        """
        return self._computes

    @property
    def is_tuning_complete(self):
        """bool: Check whether all children have completed tuning.

        ``True`` when ``is_tuning_complete`` is ``True`` for all children.

        Note:
            In MPI parallel execution, `is_tuning_complete` is ``True`` only
            when all children on **all ranks** have completed tuning.

        See Also:
            `hoomd.operation.AutotunedObject.is_tuning_complete`

        .. rubric:: Example:

        .. code-block:: python

            while (not simulation.operations.is_tuning_complete):
                simulation.run(1000)
        """
        if not self._scheduled:
            raise DataAccessError("is_tuning_complete")

        result = all(op.is_tuning_complete for op in self)
        if self._simulation.device.communicator.num_ranks == 1:
            return result
        else:
            return _hoomd.mpi_allreduce_bcast_and(
                result, self._simulation.device._cpp_exec_conf)

    def tune_kernel_parameters(self):
        """Start tuning kernel parameters in all children.

        See Also:
            `hoomd.operation.AutotunedObject.tune_kernel_parameters`

        .. rubric:: Example:

        .. code-block:: python

            simulation.operations.tune_kernel_parameters()
        """
        if not self._scheduled:
            raise RuntimeError("Call Simulation.run() before "
                               "tune_kernel_parameters.")

        for op in self:
            op.tune_kernel_parameters()

    def __getstate__(self):
        """Get the current state of the operations container for pickling."""
        # ensure that top level changes to self.__dict__ are not propagated
        state = copy(self.__dict__)
        state['_simulation'] = None
        state['_scheduled'] = False
        return state

    @property
    def _simulation(self):
        sim = self._simulation_
        if sim is not None:
            sim = sim()
            if sim is not None:
                return sim

    @_simulation.setter
    def _simulation(self, sim):
        if sim is not None:
            sim = weakref.ref(sim)
        self._simulation_ = sim
