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
from hoomd.syncedlist import SyncedList
from hoomd.operation import Analyzer, Updater, Tuner
from hoomd.typeconverter import OnlyType
from hoomd.tune import ParticleSorter


def _triggered_op_conversion(value):
    """Handle converting _TriggeredOperation to a operation, trigger pair.

    Necessary since in C++ operations do not own their trigger.
    """
    return (value._cpp_obj, value.trigger)


class Operations(Collection):
    """A mutable collection of operations which act on a `hoomd.Simulation`.

    The `Operations` class contains all the operations acting on a
    simulation. These operations are classes that perform various actions on a
    `hoomd.Simulation`. Operations can be added and removed at any point from an
    `hoomd.Operations` instance. The class provides the interface define by
    `collections.abc.Collection`. Other methods for manipulating instances
    attempt to mimic Python objects where possible, but the class is not
    simply a mutable list or set. Since there are multiple types of operations
    in HOOMD-blue,  `Operations` objects manage multiple independent
    sequences described below.

    The types of operations which can be added to an `Operations` object are
    integrators, updaters, analyzers, tuners, and computes. An `Operations` can
    only ever hold one integrator at a time. On the other hand, an `Operations`
    object can hold any number of updaters, analyzers, tuners, or computes. To
    see examples of these types of operations see `hoomd.hpmc.integrate`
    or `hoomd.md.integrate` (integrators), `hoomd.update` (updaters) ,
    `hoomd.tune` (tuners), `hoomd.dump` (analyzers), and `hoomd.md.thermo`
    (computes).

    A given instance of an operation class can only be added to a single
    `Operations` container. Likewise, a single instance cannot be added to the
    same `Operations` container more than once.

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
        self._integrator = None

        self._tuners.append(ParticleSorter())

    def add(self, operation):
        """Add operation to this container.

        Adds the provide operation to the appropriate attribute of the
        `Operations` instance.

        Args:
            operation (``operation``): A HOOMD-blue updater, analyzers, compute,
                tuner, or integrator to add to the collection.

        Raises:
            RuntimeError: raised when an operation belonging to to this or
                another `Operations` instance is passed.
            TypeError: raised when the passed operation is not of a valid type.

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
            raise RuntimeError(
                "Operation cannot be added to twice to operation lists.")
        if isinstance(operation, hoomd.integrate._BaseIntegrator):
            self.integrator = operation
            return None
        elif isinstance(operation, Tuner):
            self._tuners.append(operation)
        elif isinstance(operation, Updater):
            self._updaters.append(operation)
        elif isinstance(operation, Analyzer):
            self._analyzers.append(operation)
        else:
            raise TypeError(
                "Operation is not of the correct type to add to Operations.")

    @property
    def _sys_init(self):
        if self._simulation is None or self._simulation.state is None:
            return False
        else:
            return True

    def schedule(self):
        """Prepares all operations for a `hoomd.Simulation.run` call.

        This is provided in the public API to allow users to access quantities
        that otherwise would not be available until after a call to
        `hoomd.Simulation.run`. This does not have to be called before a ``run``
        call as this will be called by ``run`` if necessary.

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
        self._scheduled = True

    def unschedule(self):
        """Undo the effects of `Operations.schedule`."""
        self._integrator._detach()
        self._analyzers._unsync()
        self._updaters._unsync()
        self._tuners._unsync()
        self._scheduled = False

    def _store_reader(self, reader):
        # TODO
        pass

    def __contains__(self, operation):
        """Whether an operation is contained in this container.

        Args:
            operation (``any``): Returns whether this exact operation is
                contained in the collection.
        """
        return any(op is operation for op in self)

    def __iter__(self):
        """Iterates through all contained operations."""
        integrator = (self._integrator,) if self._integrator else []
        yield from chain(
            integrator, self._analyzers, self._updaters, self._tuners)

    def __len__(self):
        """Return the number of operations contained in this collection."""
        return len(list(self))

    @property
    def scheduled(self):
        """bool: Whether `Operations.schedule` has been called and is in
        effect.
        """
        return self._scheduled

    @property
    def integrator(self):
        """``Integrator``: An MD or HPMC integrator object.

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
        """list[`hoomd.operation.Analzyer`]: A list of analyzer operations.

        Holds the list of analyzers associated with this collection. The list
        can be modified as a standard Python list.
        """
        return self._analyzers

    @property
    def tuners(self):
        """list[`hoomd.operation.Tuner`]: A list of tuner operations.

        Holds the list of tuners associated with this collection. The list can be
        modified as a standard Python list.
        """
        return self._tuners

    def __iadd__(self, operation):
        """Works the same as `Operations.add`.

        Args:
            operation (``operation``): A HOOMD-blue updater, analyzers, compute,
                tuner, or integrator to add to the object.
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
            operation (``operation``): A HOOMD-blue integrator, updater,
                analyzer, tuner, or compute, to remove from the container.

        Raises:
            ValueError: raises if operation is not found in this container.
            TypeError: raises if operation is not of a valid type for this
                container.
        """
        if isinstance(operation, hoomd.integrate._BaseIntegrator):
            self.integrator = None
        elif isinstance(operation, Analyzer):
            self._analyzers.remove(operation)
        elif isinstance(operation, Updater):
            self._updaters.remove(operation)
        elif isinstance(operation, Tuner):
            self._tuners.remove(operation)
        else:
            raise TypeError(
                "operation is not a valid type for an Operations container.")

    def __isub__(self, operation):
        """Works the same as `Operations.remove`.

        Args:
            operation (``operation``): A HOOMD-blue integrator, updater,
                analyzer, tuner, or compute to remove from the collection.
        """
        self.remove(operation)
        return self
