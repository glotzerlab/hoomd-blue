import hoomd._hoomd as _hoomd
from hoomd.state import State
from hoomd.operations import Operations


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
        self._operations = Operations(self)

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
            return None
        else:
            return self._timestep

    @timestep.setter
    def timestep(self, step):
        if step < 0:
            raise ValueError("Timestep must be positive.")
        elif self._state is None:
            self._timestep = step
        else:
            raise RuntimeError("State must not be set to change timestep.")

    def create_state_from_gsd(self, filename, frame=-1):
        # initialize the system
        # Error checking
        if self.state is not None:
            raise RuntimeError("Cannot initialize more than once\n")
        filename = _hoomd.mpi_bcast_str(filename,
                                        self.device.cpp_exec_conf)
        # Grab snapshot and timestep
        reader = _hoomd.GSDReader(self.device.cpp_exec_conf,
                                  filename, abs(frame), frame < 0)
        snapshot = reader.getSnapshot()

        step = reader.getTimeStep() if self.timestep is None else self.timestep
        self._state = State(self, snapshot)

        reader.clearSnapshot()
        # Store System and Reader for Operations
        self._cpp_sys = _hoomd.System(self.state._cpp_sys_def, step)
        self.operations._store_reader(reader)

    @property
    def state(self):
        return self._state

    @property
    def operations(self):
        return self._operations

    def sanity_check(self):
        raise NotImplementedError

    def advance(self, runsteps):
        raise NotImplementedError

    def apply_operations(self, operations):
        raise NotImplementedError

    @property
    def tps(self):
        raise NotImplementedError
