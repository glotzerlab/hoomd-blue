# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

from hoomd import _hoomd
from inspect import isclass


class Trigger(_hoomd.Trigger):
    pass


class Periodic(_hoomd.PeriodicTrigger, Trigger):
    def __init__(self, period, phase=0):
        _hoomd.PeriodicTrigger.__init__(self, period, phase)

    def __str__(self):
        return f"hoomd.trigger.Periodic(period={self.period}, " \
               f"phase={self.phase}"


class Before(_hoomd.BeforeTrigger, Trigger):
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be greater than or equal to 0.")
        else:
            _hoomd.BeforeTrigger.__init__(self, timestep)

    def __str__(self):
        return f"hoomd.trigger.Before(timestep={self.timestep}"

class On(_hoomd.OnTrigger, Trigger):
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.OnTrigger.__init__(self, timestep)

    def __str__(self):
        return f"hoomd.trigger.On(timestep={self.timestep}"

class After(_hoomd.AfterTrigger, Trigger):
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.AfterTrigger.__init__(self, timestep)

    def __str__(self):
        return f"hoomd.trigger.After(timestep={self.timestep}"

class Not(_hoomd.NotTrigger, Trigger):
    def __init__(self, trigger):
        _hoomd.NotTrigger.__init__(self, trigger)

    def __str__(self):
        return f"hoomd.trigger.Not(timestep={self.timestep}"

class And(_hoomd.AndTrigger, Trigger):
    def __init__(self, triggers):
        triggers = list(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.AndTrigger.__init__(self, triggers)


class Or(_hoomd.OrTrigger, Trigger):
    def __init__(self, triggers):
        triggers = list(triggers)
        if not all(isinstance(t, Trigger) for t in triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.OrTrigger.__init__(self, triggers)
