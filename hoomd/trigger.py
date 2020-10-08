# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

""" Triggers enable users to design time points when and while 
    `hoomd.Operations` operates. The composite of triggers takes time steps and 
    returns `True` or `False`. The operation will perform when Trigger returns 
    `True`.
"""

from hoomd import _hoomd
from inspect import isclass


class Trigger(_hoomd.Trigger):  # noqa: D101
    """ Base class trigger.

    Provides methods common to all triggers.

    Attention:
        Users should instantiate the subclasses, using `Trigger` directly
        will result in an error.
    """
    pass


class Periodic(_hoomd.PeriodicTrigger, Trigger):
    """ Set periodicity of trigger.

    Args:
        period (float): timesteps for periodicity
        phase (float): timesteps for phase
    
    `hoomd.Operations` will operate when the Periodic trigger returns `True` 
    every `period` steps ``((t - phase)%period = 0)``

    Example::

            trig = hoomd.trigger.Periodic(100)
            traj_writer = hoomd.dump.GSD('simulate.gsd', 
                                        trigger=trig, 
                                        filter = hoomd.filter.All())
            csv = hoomd.output.CSV(trig, logger)

    Attributes:
        period (float): periodicity in time step.
        phase (float): phase in time step.
    """

    def __init__(self, period, phase=0):
        _hoomd.PeriodicTrigger.__init__(self, period, phase)

    def __str__(self):
        return f"hoomd.trigger.Periodic(period={self.period}, " \
               f"phase={self.phase})"


class Before(_hoomd.BeforeTrigger, Trigger):
    """ Set the timepoint to finish triger.

    Args:
        timestep (float): time step for the operation to stop working.

    `hoomd.trigger.Before` returns `True` for all time steps t < `timestep`.
    In other words, :py:class:`hoomd.Operations` will operate at less than 
    `timestep`.
    
    Example::
            
            # trigger every 100 time steps at less than first 5000 steps.
            trig = hoomd.trigger.And(
                [hoomd.trigger.Periodic(100),
                hoomd.trigger.Before(sim.timestep + 5000)])
            tune = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a','d'],
                                                        target=0.2
                                                        trigger=trig)

    Attributes:
        timestep (float): The time step for operation to stop working. 
    """
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be greater than or equal to 0.")
        else:
            _hoomd.BeforeTrigger.__init__(self, timestep)

    def __str__(self):
        return f"hoomd.trigger.Before(timestep={self.timestep})"

class On(_hoomd.OnTrigger, Trigger):
    """ Set the timepoint to trigger.

    Args:
        timestep (float): time step to trigger.

    `hoomd.trigger.On` returns `True` for time steps t = `timestep`.
    In other words, :py:class:`hoomd.Operations` will operate at `timestep`.
    
    Example::
            
            # trigger at 1000 time steps
            trig = hoomd.trigger.On(1000)
            hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a','d'],
                                                target=0.2
                                                trigger=trig)

    Attributes:
        timestep (float): time step to trigger. 
    """

    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.OnTrigger.__init__(self, timestep)

    def __str__(self):
        return f"hoomd.trigger.On(timestep={self.timestep})"

class After(_hoomd.AfterTrigger, Trigger):
    """ Set the timepoint to start trigger.

    Args:
        timestep (float): time step for the operation to start working.

    `hoomd.trigger.After` returns `True` for all time steps t > `timestep`.
    In other words, :py:class:`hoomd.Operations` will operate at greater than 
    `timestep`.

    Example::
            
            # trigger every 100 time steps after 1000 time steps.
            trig = hoomd.trigger.And([
                    hoomd.trigger.After(1000),
                    hoomd.trigger.Periodic(100)])
            hoomd.update.BoxResize(box1,box2,variant,trig)

    Attributes:
        timestep (float): The time step for operation to start working. 
    """
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.AfterTrigger.__init__(self, timestep)

    def __str__(self):
        return f"hoomd.trigger.After(timestep={self.timestep})"

class Not(_hoomd.NotTrigger, Trigger):
    """ Not operator for trigger
    Args:
        trigger (hoomd.Trigger): The trigger object to reverse.

    :py:class:`hoomd.Operations` will operate on reversed time window of `trigger`.

    Example::
            
            trig = hoomd.trigger.Not(hoomd.trigger.After(1000))
            hoomd.output.CSV(trig, logger)

    Attributes:
        trigger (hoomd.Trigger): The trigger object to reverse. 
    """
    def __init__(self, trigger):
        _hoomd.NotTrigger.__init__(self, trigger)

    def __str__(self):
        return f"hoomd.trigger.Not(trigger={self.trigger})"

class And(_hoomd.AndTrigger, Trigger):
    """ And operator for triggers

    Args:
        triggers (`list`[`hoomd.Trigger`]): List of triggers to combine

    `hoomd.trigger.And` returns `True` when all of input triggers returns `True`.
    
    Example::
            
            # trigger every 100 time steps after 1000 time steps.
            trig = hoomd.trigger.And([
                    hoomd.trigger.After(1000),
                    hoomd.trigger.Periodic(100)])
            hoomd.update.BoxResize(box1,box2,variant,trig)

    Attributes:
        triggers (List[hoomd.Trigger]): List of triggers combined
    """

    def __init__(self, triggers):
        triggers = list(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.AndTrigger.__init__(self, triggers)

    def __str__(self):
        result = "hoomd.trigger.And(["
        result += ", ".join(str(trigger) for trigger in self.triggers)
        result += "])"
        return result

class Or(_hoomd.OrTrigger, Trigger):
    """ Or operator for triggers

    Args:
        triggers (`list`[`hoomd.Trigger`]): List of triggers to combine

    `hoomd.trigger.Or` returns `True` when any of input triggers returns `True`.
    
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
            hoomd.dump.GSD('simulate.gsd', 
                           trigger=trig, 
                           filter = hoomd.filter.All())
            hoomd.output.CSV(trig, logger)

    Attributes:
        triggers (List[hoomd.Trigger]): List of triggers combined
    """
    def __init__(self, triggers):
        triggers = list(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.OrTrigger.__init__(self, triggers)

    def __str__(self):
        result = "hoomd.trigger.Or(["
        result += ", ".join(str(trigger) for trigger in self.triggers)
        result += "])"
        return result
