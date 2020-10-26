# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Triggers determine when `hoomd.Operations` activate.

A `Trigger` is a boolean valued function of the timestep. The operation will
perform its action when Trigger returns `True`. A single trigger object
may be assigned to multiple operations.

.. rubric:: User defined triggers

You can define your own triggers by subclassing `Trigger` in Python. When you do
so, override the `Trigger.compute` method and explicitly call the base class
constructor in ``__init__``.

Example:
    Define a custom trigger::

        class CustomTrigger(hoomd.trigger.Trigger):

            def __init__(self):
                hoomd.trigger.Trigger.__init__(self)

            def compute(self, timestep):
                return (timestep**(1 / 2)).is_integer()
"""

from hoomd import _hoomd


class Trigger(_hoomd.Trigger): # noqa D214
    """Base class trigger.

    Provides methods common to all triggers.

    Attention:
        Users should instantiate the subclasses, using `Trigger` directly
        will result in an error.

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
    pass


class Periodic(_hoomd.PeriodicTrigger, Trigger):
    """Trigger periodically.

    Args:
        period (int): timesteps for periodicity
        phase (int): timesteps for phase

    `hoomd.trigger.Periodic` evaluates `True` every `period` steps offset by
    phase::

        return (t - phase) % period == 0

    Example::

            trig = hoomd.trigger.Periodic(100)

    Attributes:
        period (int): periodicity in time step.
        phase (int): phase in time step.
    """

    def __init__(self, period, phase=0):
        _hoomd.PeriodicTrigger.__init__(self, period, phase)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.Periodic(period={self.period}, " \
               f"phase={self.phase})"


class Before(_hoomd.BeforeTrigger, Trigger):
    """Trigger on all steps before a given step.

    Args:
        timestep (int): The step after the trigger ends.

    `hoomd.trigger.Before` evaluates `True` for all time steps less than the
    `timestep`::

        return t < timestep

    Example::

            # trigger every 100 time steps at less than first 5000 steps.
            trigger = hoomd.trigger.And(
                [hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(sim.timestep + 5000)])

    Attributes:
        timestep (int): The step after the trigger ends.
    """
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be greater than or equal to 0.")
        else:
            _hoomd.BeforeTrigger.__init__(self, timestep)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.Before(timestep={self.timestep})"


class On(_hoomd.OnTrigger, Trigger):
    """Trigger on a specific timestep.

    Args:
        timestep (int): The timestep to trigger on.

    `hoomd.trigger.On` returns `True` for steps equal to `timestep`::

        return t == timestep

    Example::

            # trigger at 1000 time steps
            trigger = hoomd.trigger.On(1000)

    Attributes:
        timestep (int): The timestep to trigger on.
    """

    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.OnTrigger.__init__(self, timestep)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.On(timestep={self.timestep})"


class After(_hoomd.AfterTrigger, Trigger):
    """Trigger on all steps after a given step.

    Args:
        timestep (int): The step before the trigger will start.

    `hoomd.trigger.After` returns `True` for all time steps greater than
    `timestep`::

        return t > timestep

    Example::

            # trigger every 100 time steps after 1000 time steps.
            trigger = hoomd.trigger.And([
                    hoomd.trigger.After(1000),
                    hoomd.trigger.Periodic(100)])

    Attributes:
        timestep (int): The step before the trigger will start.
    """
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.AfterTrigger.__init__(self, timestep)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.After(timestep={self.timestep})"


class Not(_hoomd.NotTrigger, Trigger):
    """Negate a trigger.

    Args:
        trigger (hoomd.trigger.Trigger): The trigger object to negate.

    `hoomd.trigger.Not` returns the boolean negation of `trigger`::

        return not trigger(t)

    Example::

            trigger = hoomd.trigger.Not(hoomd.trigger.After(1000))

    Attributes:
        trigger (hoomd.trigger.Trigger): The trigger object to negate.
    """
    def __init__(self, trigger):
        _hoomd.NotTrigger.__init__(self, trigger)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        return f"hoomd.trigger.Not(trigger={self.trigger})"


class And(_hoomd.AndTrigger, Trigger):
    """Boolean and operation.

    Args:
        triggers (`list` [`hoomd.trigger.Trigger`]): List of triggers.

    `hoomd.trigger.And` returns `True` when all the input triggers returns
    `True`::

        return all([f(t) for f in triggers])

    Example::

            # trigger every 100 time steps after 1000 time steps.
            trig = hoomd.trigger.And([
                    hoomd.trigger.After(1000),
                    hoomd.trigger.Periodic(100)])

    Attributes:
        triggers (List[hoomd.trigger.Trigger]): List of triggers.
    """

    def __init__(self, triggers):
        triggers = list(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.AndTrigger.__init__(self, triggers)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        result = "hoomd.trigger.And(["
        result += ", ".join(str(trigger) for trigger in self.triggers)
        result += "])"
        return result


class Or(_hoomd.OrTrigger, Trigger):
    """Boolean or operation.

    Args:
        triggers (`list` [`hoomd.trigger.Trigger`]): List of triggers.

    `hoomd.trigger.Or` returns `True` when any of the input triggers returns
    `True`::

        return any([f(t) for f in triggers])

    Example::

            # trigger every 100 time steps before at time step of 1000.
            # or      every 10  time steps after  at time step of 1000.
            trig = hoomd.trigger.Or([hoomd.trigger.And([
                                        hoomd.trigger.Before(1000),
                                        hoomd.trigger.Periodic(100)]),
                                    [hoomd.trigger.And([
                                        hoomd.trigger.After(1000),
                                        hoomd.trigger.Periodic(10)])
                                    ])

    Attributes:
        triggers (`list` [`hoomd.trigger.Trigger`]): List of triggers.
    """
    def __init__(self, triggers):
        triggers = list(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.OrTrigger.__init__(self, triggers)

    def __str__(self):
        """Human readable representation of the trigger as a string."""
        result = "hoomd.trigger.Or(["
        result += ", ".join(str(trigger) for trigger in self.triggers)
        result += "])"
        return result
