from hoomd import _hoomd

class Trigger(_hoomd.Trigger):
    pass


class PeriodicTrigger(_hoomd.PeriodicTrigger):
    def __init__(self, period, phase=0):
        super().__init__(period, phase)
