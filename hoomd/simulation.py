import hoomd._hoomd as _hoomd
from hoomd.logging import log, Loggable
from hoomd.state import State
from hoomd.snapshot import Snapshot
from hoomd.operations import Operations
import hoomd


class Simulation(metaclass=Loggable):
    r""" Simulation.

    Args:
        device (:py:mod:`hoomd.device`): Device to execute the simulation.
    """

    def __init__(self, device):
        self._device = device
        self._state = None
        self._operations = Operations(self)
        self._verbose = False
        self._timestep = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        raise ValueError("Device cannot be removed or replaced once in "
                         "Simulation object.")

    @log
    def timestep(self):
        if not hasattr(self, '_cpp_sys'):
            return self._timestep
        else:
            return self._cpp_sys.getCurrentTimeStep()

    @timestep.setter
    def timestep(self, step):
        if step < 0:
            raise ValueError("Timestep must be positive.")
        elif self._state is None:
            self._timestep = step
        else:
            raise RuntimeError("State must not be set to change timestep.")

    def _init_communicator(self):
        """ Initialize the Communicator
        """
        # initialize communicator
        if hoomd.version.enable_mpi:
            pdata = self.state._cpp_sys_def.getParticleData()
            decomposition = pdata.getDomainDecomposition()
            if decomposition is not None:
                # create the c++ Communicator
                if isinstance(self.device, hoomd.device.CPU):
                    cpp_communicator = _hoomd.Communicator(
                        self.state._cpp_sys_def, decomposition)
                else:
                    cpp_communicator = _hoomd.CommunicatorGPU(
                        self.state._cpp_sys_def, decomposition)

                # set Communicator in C++ System
                self._cpp_sys.setCommunicator(cpp_communicator)
                self._system_communicator = cpp_communicator
            else:
                self._system_communicator = None
        else:
            self._system_communicator = None

    def create_state_from_gsd(self, filename, frame=-1):
        # initialize the system
        # Error checking
        if self.state is not None:
            raise RuntimeError("Cannot initialize more than once\n")
        filename = _hoomd.mpi_bcast_str(filename,
                                        self.device._cpp_exec_conf)
        # Grab snapshot and timestep
        reader = _hoomd.GSDReader(self.device._cpp_exec_conf,
                                  filename, abs(frame), frame < 0)
        snapshot = Snapshot._from_cpp_snapshot(reader.getSnapshot(),
                                               self.device.communicator)

        step = reader.getTimeStep() if self.timestep is None else self.timestep
        self._state = State(self, snapshot)

        reader.clearSnapshot()
        # Store System and Reader for Operations
        self._cpp_sys = _hoomd.System(self.state._cpp_sys_def, step)
        self._init_communicator()
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
        self._init_communicator()

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
    def always_compute_pressure(self):
        """Always compute the virial and pressure.

        By default, HOOMD only computes the virial and pressure on timesteps
        where it is needed (when :py:class:`hoomd.dump.GSD` writes
        log data to a file or when using an NPT integrator). Set
        ``always_compute_pressure`` to True to make the per particle virial,
        net virial, and system pressure available to query any time by property
        or through the :py:class:`hoomd.logging.Logger` interface.

        Note:
            Enabling this flag will result in a moderate performance penalty
            when using MD pair potentials.
        """
        if not hasattr(self, '_cpp_sys'):
            return False
        else:
            return self._cpp_sys.getPressureFlag()

    @always_compute_pressure.setter
    def always_compute_pressure(self, value):
        if not hasattr(self, '_cpp_sys'):
            # TODO make this work when not attached by automatically setting
            # flag when state object is instantiated.
            raise RuntimeError('Cannot set flag without state')
        else:
            self._cpp_sys.setPressureFlag(value)

            # if the flag is true, also set it in the particle data
            if value:
                self._state._cpp_sys_def.getParticleData().setPressureFlag()

    def run(self, steps):
        """Run the simulation forward a given number of steps.
        """
        # check if initialization has occurred
        if not hasattr(self, '_cpp_sys'):
            raise RuntimeError('Cannot run before state is set.')
        if not self.operations.scheduled:
            self.operations.schedule()

        self._cpp_sys.run(int(steps))
