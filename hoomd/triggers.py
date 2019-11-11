
class Trigger:
    def __init__(self):
        pass

    def __call__(self, step):
        raise NotImplementedError


class PeriodicTrigger(Trigger):
    def __init__(self, period, phase=0):
        self.period = period
        self.phase = phase

    def __call__(self, timestep):
        return (timestep - self.phase) % self.period == 0
