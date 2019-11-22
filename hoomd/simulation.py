import hoomd._hoomd as _hoomd
from hoomd.state import State
from hoomd.snapshot import Snapshot
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
        self._verbose = False

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
        snapshot = Snapshot._from_cpp_snapshot(reader.getSnapshot(),
                                               self.device.comm)

        step = reader.getTimeStep() if self.timestep is None else self.timestep
        self._state = State(self, snapshot)

        reader.clearSnapshot()
        # Store System and Reader for Operations
        self._cpp_sys = _hoomd.System(self.state._cpp_sys_def, step)
        self._cpp_sys.enableQuietRun(not self.verbose_run)
        self.operations._store_reader(reader)

    def create_state_from_snapshot(self, snapshot):
        # initialize the system
        # Error checking
        if self.state is not None:
            raise RuntimeError("Cannot initialize more than once\n")

        self._state = State(self, snapshot)

        step = 0
        if self.timestep is not None:
            step = self.timestep

        # Store System and Reader for Operations
        self._cpp_sys = _hoomd.System(self.state._cpp_sys_def, step)
        self._cpp_sys.enableQuietRun(not self.verbose_run)

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

    @property
    def verbose_run(self):
        return self._verbose

    @verbose_run.setter
    def verbose_run(self, value):
        self._verbose = bool(value)
        self._cpp_sys.enableQuietRun(not self.verbose_run)

    def run(self, tsteps):
        """Run the simulation forward tsteps."""
        # check if initialization has occurred
        if not hasattr(self, '_cpp_sys'):
            raise RuntimeError('Cannot run before state is set.')
        if not self.operations.scheduled:
            raise RuntimeError('Cannot run before operations are scheduled.')

        # if context.current.integrator is None:
        #     context.current.device.cpp_msg.warning("Starting a run without an integrator set")
        # else:
        #     context.current.integrator.update_forces()
        #     context.current.integrator.update_methods()
        #     context.current.integrator.update_thermos()

        # update all user-defined neighbor lists
        # for nl in context.current.neighbor_lists:
        #     nl.update_rcut()
        #     nl.update_exclusions_defaults()

        # detect 0 hours remaining properly
        self._cpp_sys.run(int(tsteps), 0, None, 0, 0)
