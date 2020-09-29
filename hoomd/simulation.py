# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Define the Simulation class."""

import hoomd._hoomd as _hoomd
from hoomd.logging import log, Loggable
from hoomd.state import State
from hoomd.snapshot import Snapshot
from hoomd.operations import Operations
import hoomd
import json


class Simulation(metaclass=Loggable):
    """Simulation description.

    Args:
        device (`hoomd.device.Device`): Device to execute the simulation.

    `Simulation` is the central class in HOOMD-blue that defines a simulation,
    including the `state` of the system, the `operations` that apply to the
    state during a simulation `run`, and the `device` to use when executing
    the simulation.
    """

    def __init__(self, device):
        self._device = device
        self._state = None
        self._operations = Operations(self)
        self._timestep = None

    @property
    def device(self):
        """hoomd.device._Device: Device used to execute the simulation."""
        return self._device

    @device.setter
    def device(self, value):
        raise ValueError("Device cannot be removed or replaced once in "
                         "Simulation object.")

    @log
    def timestep(self):
        """int: Current time step of the simulation.

        Note:
            Functions like `create_state_from_gsd` will set the initial timestep
            from the input. Set `timestep` before creating the simulation state
            to set the initial timestep of the simulation::

                sim.timestep = 5000
                sim.create_state_from_gsd('gsd_at_step_10000000.gsd')
                assert sim.timestep == 5000
        """
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
        """Initialize the Communicator."""
        # initialize communicator
        if hoomd.version.mpi_enabled:
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
        """Create the simulation state from a GSD file.

        Args:
            filename (str): GSD file to read

            frame (int): Index of the frame to read from the file. Negative
                values index back from the last frame in the file.
        """
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
        """Create the simulations state from a `Snapshot`.

        Args:
            snapshot (Snapshot): Snapshot to initialize the state from.

        When no timestep is provided, `create_state_from_snapshot` sets
        `timestep` to 0.

        Warning:
            *snapshot* must be a `hoomd.Snapshot`. `create_state_from_snapshot`
            does not support `gsd.hoomd.Snapshot` objects from the ``gsd``
            Python package - use `create_state_from_gsd` to read GSD files.
        """
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
        """State: The current simulation state."""
        return self._state

    @property
    def operations(self):
        """Operations: The operations that apply to the state."""
        return self._operations

    @property
    def tps(self):
        """float: The average number of time steps per second.

        `tps` resets at the start of each call to `run`. During and after the
        call `tps` is the number of steps executed divided by the elapsed
        walltime in seconds.
        """
        if self.state is None:
            return None
        else:
            return self._cpp_sys.getLastTPS()

    @property
    def always_compute_pressure(self):
        """bool: Always compute the virial and pressure (defaults to ``False``).

        By default, HOOMD only computes the virial and pressure on timesteps
        where it is needed (when :py:class:`hoomd.dump.GSD` writes
        log data to a file or when using an NPT integrator). Set
        `always_compute_pressure` to True to make the per particle virial,
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

    def run(self, steps, write_at_start=False):
        """Advance the simulation a number of steps.

        Args:
            steps (int): Number of steps to advance the simulation.

            write_at_start (bool): When True, writers
               with triggers that evaluate True for the initial step will be
               exected before the time step loop.

        Note:
            Initialize the simulation's state before calling `run`.

        During each step `run`, `Simulation` applies its `operations` to the
        state in the order: Tuners, Updaters, Integrator, then Writers following
        the logic in this pseudocode::

            if write_at_start:
                for writer in operations.writers:
                    if writer.trigger(timestep):
                        writer.write(timestep)

            end_step = timestep + steps
            while timestep < end_step:
                for tuner in operations.tuners:
                    if tuner.trigger(timestep):
                        tuner.tune(timestep)

                for updater in operations.updaters:
                    if updater.trigger(timestep):
                        updater.update(timestep)

                if operations.integrator is not None:
                    operations.integrator(timestep)

                timestep += 1

                for writer in operations.writers:
                    if writer.trigger(timestep):
                        writer.write(timestep)

        This order of operations ensures that writers (such as `hoomd.dump.GSD`)
        capture the final output of the last step of the run loop. For example,
        a writer with a trigger ``hoomd.trigger.Periodic(period=100, phase=0)``
        active during a ``run(500)`` would write on steps 100, 200, 300, 400,
        and 500. Set ``write_at_start=True`` on the first
        call to `run` to also obtain output at step 0.

        Warning:
            Using ``write_at_start=True`` in subsequent
            calls to `run` will result in duplicate output frames.
        """
        # check if initialization has occurred
        if not hasattr(self, '_cpp_sys'):
            raise RuntimeError('Cannot run before state is set.')
        if not self.operations.scheduled:
            self.operations.schedule()

        self._cpp_sys.run(int(steps), write_at_start)

    def write_debug_data(self, filename):
        """Write debug data to a JSON file.

        Args:
            filename (str): Name of file to write.

        The debug data file contains useful information for others to help you
        troubleshoot issues.

        Note:
            The file format and particular data written to this file may change
            from version to version.

        Warning:
            The specified file name will be overwritten.
        """
        debug_data = {}
        debug_data['hoomd_module'] = str(hoomd)
        debug_data['version'] = dict(
            compile_date=hoomd.version.compile_date,
            compile_flags=hoomd.version.compile_flags,
            cxx_compiler=hoomd.version.cxx_compiler,
            git_branch=hoomd.version.git_branch,
            git_sha1=hoomd.version.git_sha1,
            gpu_api_version=hoomd.version.gpu_api_version,
            gpu_enabled=hoomd.version.gpu_enabled,
            gpu_platform=hoomd.version.gpu_platform,
            install_dir=hoomd.version.install_dir,
            mpi_enabled=hoomd.version.mpi_enabled,
            source_dir=hoomd.version.source_dir,
            tbb_enabled=hoomd.version.tbb_enabled,)

        reasons = hoomd.device.GPU.get_unavailable_device_reasons()

        debug_data['device'] = dict(
            msg_file=self.device.msg_file,
            notice_level=self.device.notice_level,
            devices=self.device.devices,
            num_cpu_threads=self.device.num_cpu_threads,
            gpu_available_devices=hoomd.device.GPU.get_available_devices(),
            gpu_unavailable_device_reasons=reasons)

        debug_data['communicator'] = dict(
            num_ranks=self.device.communicator.num_ranks)

        # TODO: Domain decomposition

        if self.state is not None:
            debug_data['state'] = dict(
                types=self.state.types,
                N_particles=self.state.N_particles,
                N_bonds=self.state.N_bonds,
                N_angles=self.state.N_angles,
                N_impropers=self.state.N_impropers,
                N_special_pairs=self.state.N_special_pairs,
                N_dihedrals=self.state.N_dihedrals,
                box=repr(self.state.box))

        # save all loggable quantities from operations and their child computes
        logger = hoomd.logging.Logger(only_default=False)
        logger += self

        for op in self.operations:
            logger.add(op)

            for child in op._children:
                logger.add(child)

        log = logger.log()
        log_values = hoomd.util.dict_map(log, lambda v: v[0])
        debug_data['operations'] = log_values

        if self.device.communicator.rank == 0:
            with open(filename, 'w') as f:
                json.dump(debug_data, f, default=lambda v: str(v), indent=4)
