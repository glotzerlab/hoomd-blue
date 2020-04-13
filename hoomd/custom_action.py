from abc import ABC, abstractmethod


class _CustomAction(ABC):
    flags = []
    log_quantities = dict()

    def __init__(self):
        pass

    def attach(self, simulation):
        self._state = simulation.state

    def detach(self):
        if hasattr(self, '_state'):
            del self._state

    @abstractmethod
    def act(self, timestep):
        pass
