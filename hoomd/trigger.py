# Copyright (c) 2009-2019 The Regents of the University of Michigan
# License.
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause

from hoomd import _hoomd
from inspect import isclass


class Trigger(_hoomd.Trigger):
    pass


class Periodic(_hoomd.PeriodicTrigger, Trigger):
    def __init__(self, period, phase=0):
        _hoomd.PeriodicTrigger.__init__(self, period, phase)


class Before(_hoomd.BeforeTrigger, Trigger):
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.BeforeTrigger.__init__(self, timestep)


class On(_hoomd.OnTrigger, Trigger):
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.OnTrigger.__init__(self, timestep)


class After(_hoomd.AfterTrigger, Trigger):
    def __init__(self, timestep):
        if timestep < 0:
            raise ValueError("timestep must be positive.")
        else:
            _hoomd.AfterTrigger.__init__(self, timestep)


class Not(_hoomd.NotTrigger, Trigger):
    def __init__(self, trigger):
        _hoomd.NotTrigger.__init__(self, trigger)


class And(_hoomd.AndTrigger, Trigger):
    def __init__(self, triggers):
        if not hasattr(triggers, '__iter__') or isclass(triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.AndTrigger.__init__(self, triggers)


class Or(_hoomd.OrTrigger, Trigger):
    def __init__(self, triggers):
        if not hasattr(triggers, '__iter__') or isclass(triggers):
            raise ValueError("triggers must an iterable of Triggers.")
        _hoomd.OrTrigger.__init__(self, triggers)
