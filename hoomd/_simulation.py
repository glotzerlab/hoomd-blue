class Simulation:
    R"""
    Parameters:
        device

    Attributes:
        device
        state
        operations
    """

    def __init__(self, device):
        self._device = device
        self._state = None
        self._operations = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        raise ValueError("Device cannot be removed or replaced once in "
                         "Simulation object.")

    @property
    def timestep(self):
        if not hasattr(self, '_cpp_system'):
            return 0
        else:
            return self._timestep

    @timestep.setter
    def timestep(self, step):
        if step < 0:
            raise ValueError("Timestep must be positive.")
        elif self._state is None:
               
        else:
            raise RuntimeError("State must not be set to change timestep.")

    def create_state_from_gsd(filename, frame=-1):
        # initialize the system
        _perform_common_init_tasks()
        hoomd.context.current.state_reader = reader
        hoomd.context.current.state_reader.clearSnapshot();
        hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, time_step)

        self._system_definition
        raise NotImplementedError

    def sanity_check(self):
        raise NotImplementedError

    def advance(self, runsteps):
        raise NotImplementedError

    def apply_operations(self, operations):
        raise NotImplementedError

    @property
    def tps(self):
        raise NotImplementedError
