import hoomd._hoomd as _hoomd
from hoomd.logging import log, Loggable
from hoomd.state import State
from hoomd.snapshot import Snapshot
from hoomd.operations import Operations


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
        if _hoomd.is_MPI_available():
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
                                        self.device.cpp_exec_conf)
        # Grab snapshot and timestep
        reader = _hoomd.GSDReader(self.device.cpp_exec_conf,
                                  filename, abs(frame), frame < 0)
        snapshot = Snapshot._from_cpp_snapshot(reader.getSnapshot(),
                                               self.device.communicator)

        step = reader.getTimeStep() if self.timestep is None else self.timestep
        self._state = State(self, snapshot)

        reader.clearSnapshot()
        # Store System and Reader for Operations
        self._cpp_sys = _hoomd.System(self.state._cpp_sys_def, step)
        self._init_communicator()
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
        self._init_communicator()
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


    def run(self, tsteps):
        """Run the simulation forward tsteps."""
        # check if initialization has occurred
        if not hasattr(self, '_cpp_sys'):
            raise RuntimeError('Cannot run before state is set.')
        if not self.operations.scheduled:
            self.operations.schedule()

        # TODO either remove or refactor this code
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
