"""Implement a storage and management class for HOOMD-blue operations.

Defines the `Operations` class which serves as the main class for storing and
organizing the many parts of a simulation in a way that allows operations to be
added and removed from a `hoomd.Simulation`.
"""

# Operations also automatically handles attaching and detaching (creating and
# destroying C++ objects) for all hoomd operations.

from itertools import chain
import hoomd.integrate
from hoomd.syncedlist import SyncedList
from hoomd.operation import _Analyzer, _Updater, _Tuner
from hoomd.typeconverter import OnlyType
from hoomd.tune import ParticleSorter


def _triggered_op_conversion(value):
    """Handle converting _TriggeredOperation to a operation, trigger pair.

    Necessary since in C++ operations do not own their trigger.
    """
    return (value._cpp_obj, value.trigger)


class Operations:
    """A mutable collection of operations which act on a `hoomd.Simulation`.

    The `Operations` class serves as a holder for all the components acting on a
    simulation. These *operations* can be added and removed at any point. The
    class provides a similar interface to other Python collections where
    possible, but the class is not simply a list or set. Since there are
    multiple types of operations which HOOMD-blue has `Operations` objects
    contain multiple independent sequences.

    The types of operations which can be added to an `Operations` object are
    integrators, updaters, analyzers, tuners, and computes. An `Operations` can
    only every hold one integrator at a time. On the other hand, an `Operations`
    object can hold any number of updaters, analyzers, tuners, or computes. To
    see examples of these types of operations see (`hoomd.update`, `hoomd.tune`,
    `hoomd.dump`, and `hoomd.md.thermo`).

    Operations can only be added once to an `Operations` object. This includes
    the same object. That means to have multiple of the same operation in an
    `Operations` object, multiple instances of the operation class would need to
    be instantiated.

    Note:
        An `Operations` object is created by default when a new simulation is
        created.
    """
    def __init__(self):
        self._compute = list()
        self._scheduled = False
        self._updaters = SyncedList(OnlyType(_Updater),
                                    _triggered_op_conversion)
        self._analyzers = SyncedList(OnlyType(_Analyzer),
                                     _triggered_op_conversion)
        self._tuners = SyncedList(OnlyType(_Tuner), lambda x: x._cpp_obj)
        self._integrator = None

        self._tuners.append(ParticleSorter())

    def add(self, operation):
        """Add operation to object.

        Automatically handles adding said operation to the correct collection.
        If the operation has already been added to another `Operations` object
        including this one, then this call raises an `RuntimeError`.

        Args:
            operation (``operation``): A HOOMD-blue updater, analyzers, compute,
                tuner, or integrator to add to the object.

        Note:
            Since only one integrator can be associated with an `Operations`
            object at a time, this automatically removes the current integrator
            when called with an integrator operation.
        """
        # calling _add is handled by the synced lists and integrator property.
        # we raise this error here to provide a more clear error message.
        if operation._added:
            raise RuntimeError(
                "Operation cannot be added to twice to operation lists.")
        if isinstance(operation, hoomd.integrate._BaseIntegrator):
            self.integrator = operation
            return None
        elif isinstance(operation, _Tuner):
            self._tuners.append(operation)
        elif isinstance(operation, _Updater):
            self._updaters.append(operation)
        elif isinstance(operation, _Analyzer):
            self._analyzers.append(operation)
        else:
            raise ValueError("Operation is not of the correct type to add to"
                             " Operations.")

    @property
    def _sys_init(self):
        if self._simulation is None or self._simulation.state is None:
            return False
        else:
            return True

    def schedule(self):
        """Prepares the object to be used for a `hoomd.Simulation.run` call.

        This is provided in the public-API to allow users to get quantities that
        otherwise would not be available until after a call to
        `hoomd.Simulation.run`. This does not have to be called before a ``run``
        call as this will be called automatically if necessary.

        The function creates many C++ objects in the background to prepare for
        running a simulation. Before this all operations exist purely as Python
        objects.

        Note:
            This function will raise a `RuntimeError` when not associated with a
            `hoomd.Simulation` object.
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

    def __contains__(self, obj):
        """Whether an operation is contained in the object.

        Args:
            obj (``any``): Returns whether this exact operation is contained in
                the object.
        """
        return any(op is obj for op in self)

    def __iter__(self):
        """Iterates through all contained operations."""
        if self._integrator is not None:
            yield from chain((self._integrator,), self._analyzers,
                             self._updaters, self._tuners)
        else:
            yield from chain((self._analyzers, self._updaters, self._tuners))

    @property
    def scheduled(self):
        """Whether `Operations.schedule` has been called and is in effect."""
        return self._scheduled

    @property
    def integrator(self):
        """``Integrator``: A MD or HPMC integrator object.

        `Operations` objects have an initial ``integrator`` property of
        ``None``. Can be set to MD or HPMC integrators. The property can also be
        set to ``None``, though this will raise errors when scheduled.
        """
        return self._integrator

    @integrator.setter
    def integrator(self, op):
        if op._added:
            raise RuntimeError(
                "Integrator cannot be added to twice to Operations objects.")
        else:
            op._add(self._simulation)

        if (not isinstance(op, hoomd.integrate._BaseIntegrator)
                and op is not None):
            raise TypeError("Cannot set integrator to a type not derived "
                            "from hoomd.integrate._BaseIntegrator")
        old_ref = self.integrator
        self._integrator = op
        if self._scheduled:
            if op is not None:
                op._attach()
        if old_ref is not None:
            old_ref._notify_disconnect(self._simulation)
            old_ref._detach()
            old_ref._remove()

    @property
    def updaters(self):
        """list[``Updater``]: A list of updater operations.

        Holds the list of updaters associated with this object. The list can be
        motified as a standard Python list.
        """
        return self._updaters

    @property
    def analyzers(self):
        """list[``Analzyer``]: A list of updater operations.

        Holds the list of updaters associated with this object. The list can be
        motified as a standard Python list.
        """
        return self._analyzers

    @property
    def tuners(self):
        """list[``Tuner``]: A list of updater operations.

        Holds the list of updaters associated with this object. The list can be
        motified as a standard Python list.
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

        Checks for removal according to object id.

        Args:
            operation (``operation``): A HOOMD-blue updater, analyzers, compute,
                tuner, or integrator to remove from the object.
        """
        if isinstance(operation, hoomd.integrate._BaseIntegrator):
            raise ValueError(
                "Cannot remove iterator without setting to a new integator.")
        elif isinstance(operation, _Analyzer):
            self._analyzers.remove(operation)
        elif isinstance(operation, _Updater):
            self._updaters.remove(operation)
        elif isinstance(operation, _Tuner):
            self._tuners.remove(operation)

    def __isub__(self, operation):
        """Works the same as `Operations.remove`.

        Args:
            operation (``operation``): A HOOMD-blue updater, analyzers, compute,
                tuner, or integrator to remove from the object.
        """
        self.remove(operation)
        return self
