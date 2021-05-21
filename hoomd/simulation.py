# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Define the Simulation class."""
import inspect

import hoomd._hoomd as _hoomd
from hoomd.logging import log, Loggable
from hoomd.state import State
from hoomd.snapshot import Snapshot
from hoomd.operations import Operations
import hoomd
import json

TIMESTEP_MAX = 2**64 - 1
SEED_MAX = 2**16 - 1


class Simulation(metaclass=Loggable):
    """Define a simulation.

    Args:
        device (`hoomd.device.Device`): Device to execute the simulation.
        seed (int): Random number seed.

    `Simulation` is the central class in HOOMD-blue that defines a simulation,
    including the `state` of the system, the `operations` that apply to the
    state during a simulation `run`, and the `device` to use when executing
    the simulation.

    `seed` sets the seed for the random number generator used by all operations
    added to this `Simulation`.
    """

    def __init__(self, device, seed=None):
        self._device = device
        self._state = None
        self._operations = Operations()
        self._operations._simulation = self
        self._timestep = None
        self._seed = seed

    @property
    def device(self):
        """hoomd.device.Device: Device used to execute the simulation."""
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
            to override values from ``create_`` methods::

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
        if int(step) < 0 or int(step) > TIMESTEP_MAX:
            raise ValueError(f"steps must be in the range [0, {TIMESTEP_MAX}]")
        elif self._state is None:
            self._timestep = step
        else:
            raise RuntimeError("State must not be set to change timestep.")

    @log
    def seed(self):
        """int: Random number seed.

        Seeds are in the range [0, 65535]. When set, `seed` will take only the
        lowest 16 bits of the given value.

        HOOMD-blue uses a deterministic counter based pseudorandom number
        generator. Any time a random value is needed, HOOMD-blue computes it as
        a function of the user provided seed `seed` (16 bits), the current
        `timestep` (lower 40 bits), particle identifiers, MPI ranks, and other
        unique identifying values as needed to sample uncorrelated values:
        ``random_value = f(seed, timestep, ...)``
        """
        if self.state is None or self._seed is None:
            return self._seed
        else:
            return self._state._cpp_sys_def.getSeed()

    @seed.setter
    def seed(self, v):
        v_int = int(v)
        if v_int < 0 or v_int > SEED_MAX:
            v_int = v_int & SEED_MAX
            self.device._cpp_msg.warning(
                f"Provided seed {v} is larger than {SEED_MAX}. "
                f"Truncating to {v_int}.\n")

        self._seed = v_int

        if self._state is not None:
            self._state._cpp_sys_def.setSeed(v_int)

    def _init_system(self, step):
        """Initialize the system State.

        Perform additional initialization operations not in the State
        constructor.
        """
        self._cpp_sys = _hoomd.System(self.state._cpp_sys_def, step)

        if self._seed is not None:
            self._state._cpp_sys_def.setSeed(self._seed)

        self._init_communicator()

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

    def _warn_if_seed_unset(self):
        if self.seed is None:
            self.device._cpp_msg.warning(
                "Simulation.seed is not set, using default seed=0\n")

    def create_state_from_gsd(self, filename, frame=-1):
        """Create the simulation state from a GSD file.

        Args:
            filename (str): GSD file to read

            frame (int): Index of the frame to read from the file. Negative
                values index back from the last frame in the file.
        """
        if self.state is not None:
            raise RuntimeError("Cannot initialize more than once\n")
        filename = _hoomd.mpi_bcast_str(filename, self.device._cpp_exec_conf)
        # Grab snapshot and timestep
        reader = _hoomd.GSDReader(self.device._cpp_exec_conf, filename,
                                  abs(frame), frame < 0)
        snapshot = Snapshot._from_cpp_snapshot(reader.getSnapshot(),
                                               self.device.communicator)

        step = reader.getTimeStep() if self.timestep is None else self.timestep
        self._state = State(self, snapshot)

        reader.clearSnapshot()

        self._init_system(step)

    def create_state_from_snapshot(self, snapshot):
        """Create the simulations state from a `Snapshot`.

        Args:
            snapshot (Snapshot or gsd.hoomd.Snapshot): Snapshot to initialize
                the state from. A `gsd.hoomd.Snapshot` will first be
                converted to a `hoomd.Snapshot`.


        When `timestep` is `None` before calling, `create_state_from_snapshot`
        sets `timestep` to 0.
        """
        if self.state is not None:
            raise RuntimeError("Cannot initialize more than once\n")

        if isinstance(snapshot, Snapshot):
            # snapshot is hoomd.Snapshot
            self._state = State(self, snapshot)
        elif _match_class_path(snapshot, 'gsd.hoomd.Snapshot'):
            # snapshot is gsd.hoomd.Snapshot
            snapshot = Snapshot.from_gsd_snapshot(snapshot,
                                                  self._device.communicator)
            self._state = State(self, snapshot)
        else:
            raise TypeError(
                "Snapshot must be a hoomd.Snapshot or gsd.hoomd.Snapshot.")

        step = 0
        if self.timestep is not None:
            step = self.timestep

        self._init_system(step)

    @property
    def state(self):
        """hoomd.State: The current simulation state."""
        return self._state

    @property
    def operations(self):
        """hoomd.Operations: The operations that apply to the state."""
        return self._operations

    @operations.setter
    def operations(self, operations):
        # This condition is necessary to allow for += and -= operators to work
        # correctly with simulation.operations (+=/-=).
        if operations is self._operations:
            return
        else:
            # Handle error cases first
            if operations._scheduled or operations._simulation is not None:
                raise RuntimeError(
                    "Cannot add `hoomd.Operations` object that belongs to "
                    "another `hoomd.Simulation` object.")
            # Switch out `hoomd.Operations` objects.
            reschedule = False
            if self._operations._scheduled:
                self._operations._unschedule()
                reschedule = True

            self._operations._simulation = None
            operations._simulation = self
            self._operations = operations

            if reschedule:
                self._operations._schedule()

    @log
    def tps(self):
        """float: The average number of time steps per second.

        `tps` is the number of steps executed divided by the elapsed
        walltime in seconds. It is updated during the `run` loop and remains
        fixed after `run` completes.

        Note:
            The start time and step are reset at the beginning of each call to
            `run`.
        """
        if self.state is None:
            return None
        else:
            return self._cpp_sys.getLastTPS()

    @log
    def walltime(self):
        """float: The walltime spent during the last call to `run`.

        `walltime` is the number seconds that the last call to `run` took to
        complete. It is updated during the `run` loop and remains fixed after
        `run` completes.

        Note:
            `walltime` resets to 0 at the beginning of each call to `run`.
        """
        if self.state is None:
            return 0
        else:
            return self._cpp_sys.walltime

    @log
    def final_timestep(self):
        """float: `run` will end at this timestep.

        `final_timestep` is the timestep on which the currently executing `run`
        will complete.
        """
        if self.state is None:
            return self.timestep
        else:
            return self._cpp_sys.final_timestep

    @property
    def always_compute_pressure(self):
        """bool: Always compute the virial and pressure (defaults to ``False``).

        By default, HOOMD only computes the virial and pressure on timesteps
        where it is needed (when :py:class:`hoomd.write.GSD` writes
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

            write_at_start (bool): When `True`, writers
               with triggers that evaluate `True` for the initial step will be
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

        This order of operations ensures that writers (such as
        `hoomd.write.GSD`) capture the final output of the last step of the run
        loop. For example, a writer with a trigger
        ``hoomd.trigger.Periodic(period=100, phase=0)`` active during a
        ``run(500)`` would write on steps 100, 200, 300, 400, and 500. Set
        ``write_at_start=True`` on the first call to `run` to also obtain output
        at step 0.

        Warning:
            Using ``write_at_start=True`` in subsequent
            calls to `run` will result in duplicate output frames.
        """
        # check if initialization has occurred
        if not hasattr(self, '_cpp_sys'):
            raise RuntimeError('Cannot run before state is set.')
        if self._state._in_context_manager:
            raise RuntimeError(
                "Cannot call run inside of a local snapshot context manager.")
        if not self.operations._scheduled:
            self.operations._schedule()

        steps_int = int(steps)
        if steps_int < 0 or steps_int > TIMESTEP_MAX - 1:
            raise ValueError(f"steps must be in the range [0, "
                             f"{TIMESTEP_MAX-1}]")

        self._cpp_sys.run(steps_int, write_at_start)

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
            tbb_enabled=hoomd.version.tbb_enabled,
        )

        reasons = hoomd.device.GPU.get_unavailable_device_reasons()

        debug_data['device'] = dict(
            msg_file=self.device.msg_file,
            notice_level=self.device.notice_level,
            devices=self.device.devices,
            num_cpu_threads=self.device.num_cpu_threads,
            gpu_available_devices=hoomd.device.GPU.get_available_devices(),
            gpu_unavailable_device_reasons=reasons)

        debug_data['communicator'] = dict(
            num_ranks=self.device.communicator.num_ranks,
            partition=self.device.communicator.partition)

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


def _match_class_path(obj, *matches):
    return any(cls.__module__ + '.' + cls.__name__ in matches
               for cls in inspect.getmro(type(obj)))
