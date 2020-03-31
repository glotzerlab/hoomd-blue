# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

from hoomd import _hoomd


class Trigger(_hoomd.Trigger):
    pass


class Periodic(_hoomd.PeriodicTrigger, Trigger):
    def __init__(self, period, phase=0):
        _hoomd.PeriodicTrigger.__init__(self, period, phase)


class Until(_hoomd.UntilTrigger, Trigger):
    def __init__(self, until):
        if until < 0:
            raise ValueError("until must be positive.")
        else:
            _hoomd.UntilTrigger.__init__(self, until)


class After(_hoomd.AfterTrigger, Trigger):
    def __init__(self, after):
        if after < 0:
            raise ValueError("after must be positive.")
        else:
            _hoomd.AfterTrigger.__init__(self, after)


class Not(_hoomd.NotTrigger, Trigger):
    def __init__(self, trigger):
        _hoomd.NotTrigger.__init__(self, trigger)


class And(_hoomd.AndTrigger, Trigger):
    def __init__(self, trigger1, trigger2):
        _hoomd.AndTrigger.__init__(self, trigger1, trigger2)


class Or(_hoomd.OrTrigger, Trigger):
    def __init__(self, trigger1, trigger2):
        _hoomd.OrTrigger.__init__(self, trigger1, trigger2)
