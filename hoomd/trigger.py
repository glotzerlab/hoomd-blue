# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Triggers determine when most `hoomd.operation.Operation` instances activate.

A `Trigger` is a boolean valued function of the timestep. The operation will
perform its action when Trigger returns `True`. A single trigger object
may be assigned to multiple operations.

See `Trigger` for details on creating user-defined triggers or use one of the
provided subclasses.

.. invisible-code-block: python

    other_trigger = hoomd.trigger.Periodic(period=100)
    other_trigger1 = hoomd.trigger.Periodic(period=100)
    other_trigger2 = hoomd.trigger.Periodic(period=100)
"""

import typing

from hoomd import _hoomd

# Note: We use pybind11's pickling infrastructure for simple triggers like
# Periodic, Before, After, and On. However, we use __reduce__ for classes with
# that are composed by other triggers. We do this not because we can't do this
# in pybind11, we can, but because we would upon unpickling downcast the
# triggers to their pybind11 defined type. This happens as all references to the
# composing triggers besides the C++ class are gone, the Python garbage
# collector removes them. If we store the triggers on the Python side as well,
# since __init__ is not called in unpickling, the attributes are not
# initialized if we use pybind11 pickling.

# For the base class Trigger, we also create __getstate__ and __setstate__
# methods which should allow for pickling Python subclasses. pybind11's
# facilities do not work as they prevent us from getting the attributes of the
# class be pickled and unpickled. We manual pass and set the instance __dict__
# instead and instantiate _hoomd.Trigger in __setstate__ (which has not already
# been called as __init__ was not called).


class Trigger(_hoomd.Trigger):
    """Base class trigger.

    Provides methods common to all triggers and a base class for user-defined
    triggers.

    Subclasses should override the `Trigger.compute` method and must explicitly
    call the base class constructor in ``__init__``:

    .. code-block:: python

        class CustomTrigger(hoomd.trigger.Trigger):

            def __init__(self):
                hoomd.trigger.Trigger.__init__(self)

            def compute(self, timestep):
                return (timestep**(1 / 2)).is_integer()

    Methods:
        __call__(timestep):
            Evaluate the trigger.

            Args:
                timestep (int): The timestep.

            Note:
                When called several times with the same *timestep*, `__call__`
                calls `compute` on the first invocation, caches the value,
                and returns that cached value in subsequent calls.

            Returns:
                bool: `True` when the trigger is active, `False` when it is not.

        compute(timestep):
            Evaluate the trigger.

            Args:
                timestep (int): The timestep.

            Returns:
                bool: `True` when the trigger is active, `False` when it is not.
    """

    def __getstate__(self):
        """Get the state of the trigger object."""
        return self.__dict__

    def __setstate__(self, state):
        """Set the state of the trigger object."""
        _hoomd.Trigger.__init__(self)
        self.__dict__ = state


class Periodic(_hoomd.PeriodicTrigger, Trigger):
    """Trigger periodically.

    Args:
        period (int): timesteps for periodicity
        phase (int): timesteps for phase

    `Periodic` evaluates `True` every `period` steps offset by phase::

        return (t - phase) % period == 0

    .. rubric:: Example:

    .. code-block:: python

            trigger = hoomd.trigger.Periodic(period=100)

    Attributes:
        period (int): periodicity in time step.
        phase (int): phase in time step.
    """

    def __init__(self, period, phase=0):
        Trigger.__init__(self)
        _hoomd.PeriodicTrigger.__init__(self, period, phase)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.Periodic(period={self.period}, " \
               f"phase={self.phase})"

    def __eq__(self, other):
        """Test for equivalent triggers."""
        return (isinstance(other, Periodic) and self.period == other.period
                and self.phase == other.phase)


class Before(_hoomd.BeforeTrigger, Trigger):
    """Trigger on all steps before a given step.

    Args:
        timestep (int): The step after the trigger ends.

    `Before` evaluates `True` for all time steps less than the `timestep`::

        return t < timestep

    .. rubric:: Example:

    .. code-block:: python

            trigger = hoomd.trigger.Before(5000)

    Attributes:
        timestep (int): The step after the trigger ends.
    """

    def __init__(self, timestep):
        Trigger.__init__(self)
        if timestep < 0:
            raise ValueError("timestep must be greater than or equal to 0.")
        else:
            _hoomd.BeforeTrigger.__init__(self, timestep)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.Before(timestep={self.timestep})"

    def __eq__(self, other):
        """Test for equivalent triggers."""
        return isinstance(other, Before) and self.timestep == other.timestep


class On(_hoomd.OnTrigger, Trigger):
    """Trigger on a specific timestep.

    Args:
        timestep (int): The timestep to trigger on.

    `On` returns `True` for steps equal to `timestep`::

        return t == timestep

    .. rubric:: Example:

    .. code-block:: python

            trigger = hoomd.trigger.On(1000)

    Attributes:
        timestep (int): The timestep to trigger on.
    """

    def __init__(self, timestep):
        Trigger.__init__(self)
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.OnTrigger.__init__(self, timestep)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.On(timestep={self.timestep})"

    def __eq__(self, other):
        """Test for equivalent triggers."""
        return isinstance(other, On) and self.timestep == other.timestep


class After(_hoomd.AfterTrigger, Trigger):
    """Trigger on all steps after a given step.

    Args:
        timestep (int): The step before the trigger will start.

    `After` returns `True` for all time steps greater than `timestep`::

        return t > timestep

    .. rubric:: Example:

    .. code-block:: python

            trigger = hoomd.trigger.After(1000)

    Attributes:
        timestep (int): The step before the trigger will start.
    """

    def __init__(self, timestep):
        Trigger.__init__(self)
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.AfterTrigger.__init__(self, timestep)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.After(timestep={self.timestep})"

    def __eq__(self, other):
        """Test for equivalent triggers."""
        return isinstance(other, After) and self.timestep == other.timestep


class Not(_hoomd.NotTrigger, Trigger):
    """Negate a trigger.

    Args:
        trigger (hoomd.trigger.Trigger): The trigger object to negate.

    `Not` returns the boolean negation of `trigger`::

        return not trigger(t)

    .. rubric:: Example:

    .. code-block:: python

            trigger = hoomd.trigger.Not(other_trigger)

    Attributes:
        trigger (hoomd.trigger.Trigger): The trigger object to negate.
    """

    def __init__(self, trigger):
        Trigger.__init__(self)
        _hoomd.NotTrigger.__init__(self, trigger)
        self._trigger = trigger

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.Not(trigger={self.trigger})"

    @property
    def trigger(self):  # noqa: D102 - documented in Attributes above
        return self._trigger

    def __reduce__(self):
        """Format trigger for pickling."""
        return (type(self), (self._trigger,))

    def __eq__(self, other):
        """Test for equivalent triggers."""
        return isinstance(other, Not) and self.trigger == other.trigger


class And(_hoomd.AndTrigger, Trigger):
    """Boolean and operation.

    Args:
        triggers (`list` [`Trigger`]): List of triggers.

    `And` returns `True` when all the input triggers returns `True`::

        return all([f(t) for f in triggers])

    .. rubric:: Example:

    .. code-block:: python

            trigger = hoomd.trigger.And([other_trigger1, other_trigger2])

    Attributes:
        triggers (list[hoomd.trigger.Trigger]): List of triggers.
    """

    def __init__(self, triggers):
        Trigger.__init__(self)
        triggers = tuple(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.AndTrigger.__init__(self, triggers)
        self._triggers = triggers

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        result = "hoomd.trigger.And(["
        result += ", ".join(str(trigger) for trigger in self.triggers)
        result += "])"
        return result

    @property
    def triggers(self):  # noqa: D102 - documented in Attributes above
        return self._triggers

    def __reduce__(self):
        """Format trigger for pickling."""
        return (type(self), (self._triggers,))

    def __eq__(self, other):
        """Test for equivalent triggers."""
        return isinstance(other, And) and self.triggers == other.triggers


class Or(_hoomd.OrTrigger, Trigger):
    """Boolean or operation.

    Args:
        triggers (`list` [`Trigger`]): List of triggers.

    `Or` returns `True` when any of the input triggers returns `True`::

        return any([f(t) for f in triggers])

    .. rubric:: Example:

    .. code-block:: python

            trig = hoomd.trigger.Or([other_trigger1, other_trigger2])

    Attributes:
        triggers (`list` [`Trigger`]): List of triggers.
    """

    def __init__(self, triggers):
        Trigger.__init__(self)
        triggers = tuple(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.OrTrigger.__init__(self, triggers)
        self._triggers = triggers

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        result = "hoomd.trigger.Or(["
        result += ", ".join(str(trigger) for trigger in self.triggers)
        result += "])"
        return result

    @property
    def triggers(self):  # noqa: D102 - documented in Attributes above
        return self._triggers

    def __reduce__(self):
        """Format trigger for pickling."""
        return (type(self), (self._triggers,))

    def __eq__(self, other):
        """Test for equivalent triggers."""
        return isinstance(other, Or) and self.triggers == other.triggers


trigger_like = typing.Union[Trigger, int]
"""
An object that can serve as a trigger for an operation.

Any instance of a `Trigger` subclass is allowed, as well as an int instance or
any object convertible to an int. The integer is converted to a `Periodic`
trigger via ``Periodic(period=int(a))`` where ``a`` is the passed integer.

Note:
    Attributes that are `Trigger` objects can be set via a `trigger_like`
    object.
"""
