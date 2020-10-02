"""Implement a storage and management class for HOOMD-blue operations.

Defines the `Operations` class which serves as the main class for storing and
organizing the many parts of a simulation in a way that allows operations to be
added and removed from a `hoomd.Simulation`.
"""

# Operations also automatically handles attaching and detaching (creating and
# destroying C++ objects) for all hoomd operations.

from collections.abc import Collection
from itertools import chain
import hoomd.integrate
from hoomd.data.syncedlist import SyncedList
from hoomd.data.typeconverter import OnlyType
from hoomd.operation import Analyzer, Updater, Tuner, Compute
from hoomd.tune import ParticleSorter


def _triggered_op_conversion(value):
    """Convert _TriggeredOperation to a operation, trigger pair.

    Necessary since in C++ operations do not own their trigger.
    """
    return (value._cpp_obj, value.trigger)


class Operations(Collection):
    """A mutable collection of operations which act on a `hoomd.Simulation`.

    The `Operations` class contains all the operations acting on a
    simulation. These operations are classes that perform various actions on a
    `hoomd.Simulation`. Operations can be added and removed at any point from a
    `hoomd.Operations` instance. The class provides the interface defined by
    `collections.abc.Collection`. Other methods for manipulating instances
    attempt to mimic Python objects where possible, but the class is not
    simply a mutable list or set. Since there are multiple types of operations
    in HOOMD-blue, `Operations` objects manage multiple independent
    sequences described below.

    The types of operations which can be added to an `Operations` object are
    tuners, updaters, integrators, analyzers, and computes. An `Operations` can
    only ever hold one integrator at a time. On the other hand, an `Operations`
    object can hold any number of tuners, updaters, analyzers, or computes. To
    see examples of these types of operations see `hoomd.tune` (tuners),
    `hoomd.update` (updaters), `hoomd.hpmc.integrate` or `hoomd.md.integrate`
    (integrators), , `hoomd.dump` (analyzers), and `hoomd.md.thermo`
    (computes).

    A given instance of an operation class can only be added to a single
    `Operations` container. Likewise, a single instance cannot be added to the
    same `Operations` container more than once.

    All `Operations` instances start with a `hoomd.tune.ParticleSorter` instance
    in their ``tuners`` attribute. This is vital for increasing simulation
    performance. However, users can choose to modify or remove this tuner if
    desired.

    Note:
        An `Operations` object is created by default when a new simulation is
        created.
    """

    def __init__(self):
        self._compute = list()
        self._scheduled = False
        self._updaters = SyncedList(OnlyType(Updater),
                                    _triggered_op_conversion)
        self._analyzers = SyncedList(OnlyType(Analyzer),
                                     _triggered_op_conversion)
        self._tuners = SyncedList(OnlyType(Tuner), lambda x: x._cpp_obj)
        self._computes = SyncedList(OnlyType(Compute), lambda x: x._cpp_obj)
        self._integrator = None

        self._tuners.append(ParticleSorter())

    def _get_proper_container(self, operation):
        if isinstance(operation, Updater):
            return self._updaters
        elif isinstance(operation, Analyzer):
            return self._analyzers
        elif isinstance(operation, Tuner):
            return self._tuners
        elif isinstance(operation, Compute):
            return self._computes
        else:
            raise TypeError(
                f"{type(operation)} is not a valid operation type.")

    def add(self, operation):
        """Add operation to this container.

        Adds the provide operation to the appropriate attribute of the
        `Operations` instance.

        Args:
            operation (`hoomd.operation._Operation`): A HOOMD-blue tuner,
                updater, integrator, analyzer, or compute,  to add to the
                collection.

        Raises:
            ValueError: If ``operation`` already belongs to this or another
                `Operations` instance.
            TypeError: If ``operation`` is not of a valid type.

        Note:
            Since only one integrator can be associated with an `Operations`
            object at a time, this removes the current integrator when called
            with an integrator operation. Also, the ``integrator`` property
            cannot be set to ``None`` using this function. Use
            ``operations.integrator = None`` explicitly for this.
        """
        # calling _add is handled by the synced lists and integrator property.
        # we raise this error here to provide a more clear error message.
        if operation._added:
            raise ValueError("The provided operation has already been added "
                             "to an Operations instance.")
        if isinstance(operation, hoomd.integrate._BaseIntegrator):
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
            operation (`hoomd.operation._Operation`): A HOOMD-blue tuner,
                updater, integrator, analyzer, or compute to add to the object.
        """
        self.add(operation)
        return self

    def remove(self, operation):
        """Remove an operation from the `Operations` object.

        Remove the item from the collection whose id is the same as
        ``operation``. See
        `<https://docs.python.org/3/library/functions.html#id>`_ for the concept
        of a Python object id.

        Args:
            operation (`hoomd.operation._Operation`): A HOOMD-blue integrator,
                tuner, updater, integrator, or compute to remove from the
                container.

        Raises:
            ValueError: If ``operation`` is not found in this container.
            TypeError: If ``operation`` is not of a valid type.
        """
        if isinstance(operation, hoomd.integrate._BaseIntegrator):
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
            operation (`hoomd.operation._Operation`): A HOOMD-blue integrator,
                tuner, updater, integrator, analzyer, or compute to remove from
                the collection.
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
            self.integrator._attach()
        if not self.updaters._synced:
            self.updaters._sync(sim, sim._cpp_sys.updaters)
        if not self.analyzers._synced:
            self.analyzers._sync(sim, sim._cpp_sys.analyzers)
        if not self.tuners._synced:
            self.tuners._sync(sim, sim._cpp_sys.tuners)
        if not self.computes._synced:
            self.computes._sync(sim, sim._cpp_sys.computes)
        self._scheduled = True

    def _unschedule(self):
        """Undo the effects of `Operations._schedule`."""
        self._integrator._detach()
        self._analyzers._unsync()
        self._updaters._unsync()
        self._tuners._unsync()
        self._computes._unsync()
        self._scheduled = False

    def _store_reader(self, reader):
        # TODO
        pass

    def __contains__(self, operation):
        """Whether an operation is contained in this container.

        Args:
            operation: Returns whether this exact operation is
                contained in the collection.
        """
        return any(op is operation for op in self)

    def __iter__(self):
        """Iterates through all contained operations."""
        integrator = (self._integrator,) if self._integrator else []
        yield from chain(
            self._tuners, self._updaters, integrator, self._analyzers,
            self._computes)

    def __len__(self):
        """Return the number of operations contained in this collection."""
        return len(list(self))

    @property
    def integrator(self):
        """`hoomd.integrate._BaseIntegrator`: An MD or HPMC integrator object.

        `Operations` objects have an initial ``integrator`` property of
        ``None``. Can be set to MD or HPMC integrators. The property can also be
        set to ``None``.
        """
        return self._integrator

    @integrator.setter
    def integrator(self, op):
        if op is not None:
            if not isinstance(op, hoomd.integrate._BaseIntegrator):
                raise TypeError("Cannot set integrator to a type not derived "
                                "from hoomd.integrate._BaseIntegrator")
            if op._added:
                raise RuntimeError("Integrator cannot be added to twice to "
                                   "Operations collection.")
            else:
                op._add(self._simulation)

        old_ref = self.integrator
        self._integrator = op
        # Handle attaching and detaching integrators dealing with None values
        if self._scheduled:
            if op is not None:
                op._attach()
        if old_ref is not None:
            old_ref._notify_disconnect(self._simulation)
            old_ref._detach()
            old_ref._remove()

    @property
    def updaters(self):
        """list[`hoomd.operation.Updater`]: A list of updater operations.

        Holds the list of updaters associated with this collection. The list can
        be modified as a standard Python list.
        """
        return self._updaters

    @property
    def analyzers(self):
        """list[`hoomd.operation.Analyzer`]: A list of analyzer operations.

        Holds the list of analyzers associated with this collection. The list
        can be modified as a standard Python list.
        """
        return self._analyzers

    @property
    def tuners(self):
        """list[`hoomd.operation.Tuner`]: A list of tuner operations.

        Holds the list of tuners associated with this collection. The list can
        be modified as a standard Python list.
        """
        return self._tuners

    @property
    def computes(self):
        """list[`hoomd.operation.Compute`]: A list of tuner operations.

        Holds the list of tuners associated with this collection. The list can
        be modified as a standard Python list.
        """
        return self._computes
