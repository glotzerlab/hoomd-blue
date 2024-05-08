# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define the Simulation class.

.. invisible-code-block: python

    # prepare snapshot and gsd file for later examples
    path = tmp_path
    simulation = hoomd.util.make_example_simulation()

    snapshot = simulation.state.get_snapshot()
    gsd_filename = path / 'file.gsd'
    hoomd.write.GSD.write(state=simulation.state,
                          filename = gsd_filename,
                          filter=hoomd.filter.All())

    logger = hoomd.logging.Logger()
"""
import inspect

import hoomd._hoomd as _hoomd
from hoomd.logging import log, Loggable
from hoomd.state import State
from hoomd.snapshot import Snapshot
from hoomd.operations import Operations
import hoomd

TIMESTEP_MAX = 2**64 - 1
SEED_MAX = 2**16 - 1


class Simulation(metaclass=Loggable):
    """Define a simulation.

    Args:
        device (hoomd.device.Device): Device to execute the simulation.
        seed (int): Random number seed.

    `Simulation` is the central class that defines a simulation, including the
    `state` of the system, the `operations` that apply to the state during a
    simulation `run`, and the `device` to use when executing the simulation.

    `seed` sets the seed for the random number generator used by all operations
    added to this `Simulation`.

    Newly initialized `Simulation` objects have no state. Call
    `create_state_from_gsd` or `create_state_from_snapshot` to initialize the
    simulation's `state`.

    .. rubric:: Example:

    .. code-block:: python

        simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    """

    def __init__(self, device, seed=None):
        self._device = device
        self._state = None
        self._operations = Operations()
        self._operations._simulation = self
        self._timestep = None
        self._seed = None
        if seed is not None:
            self.seed = seed

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
        """int: The current simulation time step.

        `timestep` is read only after creating the simulation state.

        Note:
            Methods like `create_state_from_gsd` will set the initial timestep
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

        .. rubric:: Example:

        .. code-block:: python

            simulation.seed = 2
        """
        if self._state is None or self._seed is None:
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

                # set Communicator in C++ System and SystemDefinition
                self._cpp_sys.setCommunicator(cpp_communicator)
                self.state._cpp_sys_def.setCommunicator(cpp_communicator)
                self._system_communicator = cpp_communicator
            else:
                self._system_communicator = None
        else:
            self._system_communicator = None

    def _warn_if_seed_unset(self):
        if self.seed is None:
            self.device._cpp_msg.warning(
                "Simulation.seed is not set, using default seed=0\n")

    def create_state_from_gsd(self,
                              filename,
                              frame=-1,
                              domain_decomposition=(None, None, None)):
        """Create the simulation state from a GSD file.

        Args:
            filename (str): GSD file to read

            frame (int): Index of the frame to read from the file. Negative
                values index back from the last frame in the file.

            domain_decomposition (tuple): Choose how to distribute the state
                across MPI ranks with domain decomposition. Provide a tuple
                of 3 integers indicating the number of evenly spaced domains in
                the x, y, and z directions (e.g. ``(8,4,2)``). Provide a tuple
                of 3 lists of floats to set the fraction of the simulation box
                to include in each domain. The sum of each list of floats must
                be 1.0 (e.g. ``([0.25, 0.75], [0.2, 0.8], [1.0])``).

        When `timestep` is `None` before calling, `create_state_from_gsd`
        sets `timestep` to the value in the selected GSD frame in the file.

        Note:
            Set any or all of the ``domain_decomposition`` tuple elements to
            `None` and `create_state_from_gsd` will select a value that
            minimizes the surface area between the domains (e.g.
            ``(2,None,None)``). The domains are spaced evenly along each
            automatically selected direction. The default value of ``(None,
            None, None)`` will automatically select the number of domains in all
            directions.

        .. rubric:: Example:

        .. code-block:: python

            simulation.create_state_from_gsd(filename=gsd_filename)
        """
        if self._state is not None:
            raise RuntimeError("Cannot initialize more than once\n")
        filename = _hoomd.mpi_bcast_str(filename, self.device._cpp_exec_conf)
        # Grab snapshot and timestep
        reader = _hoomd.GSDReader(self.device._cpp_exec_conf, filename,
                                  abs(frame), frame < 0)
        snapshot = Snapshot._from_cpp_snapshot(reader.getSnapshot(),
                                               self.device.communicator)

        step = reader.getTimeStep() if self.timestep is None else self.timestep
        self._state = State(self, snapshot, domain_decomposition)

        reader.clearSnapshot()

        self._init_system(step)

    def create_state_from_snapshot(self,
                                   snapshot,
                                   domain_decomposition=(None, None, None)):
        """Create the simulation state from a `Snapshot`.

        Args:
            snapshot (Snapshot or gsd.hoomd.Frame): Snapshot to initialize
                the state from. A `gsd.hoomd.Frame` will first be
                converted to a `hoomd.Snapshot`.

            domain_decomposition (tuple): Choose how to distribute the state
                across MPI ranks with domain decomposition. Provide a tuple
                of 3 integers indicating the number of evenly spaced domains in
                the x, y, and z directions (e.g. ``(8,4,2)``). Provide a tuple
                of 3 lists of floats to set the fraction of the simulation box
                to include in each domain. The sum of each list of floats must
                be 1.0 (e.g. ``([0.25, 0.75], [0.2, 0.8], [1.0])``).

        When `timestep` is `None` before calling, `create_state_from_snapshot`
        sets `timestep` to 0.

        Note:
            Set any or all of the ``domain_decomposition`` tuple elements to
            `None` and `create_state_from_snapshot` will select a value that
            minimizes the surface area between the domains (e.g.
            ``(2,None,None)``). The domains are spaced evenly along each
            automatically selected direction. The default value of ``(None,
            None, None)`` will automatically select the number of domains in all
            directions.

        See Also:
            `State.get_snapshot`

            `State.set_snapshot`

        .. rubric:: Example:

        .. invisible-code-block: python

            simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)

        .. code-block:: python

            simulation.create_state_from_snapshot(snapshot=snapshot)
        """
        if self._state is not None:
            raise RuntimeError("Cannot initialize more than once\n")

        if isinstance(snapshot, Snapshot):
            # snapshot is hoomd.Snapshot
            self._state = State(self, snapshot, domain_decomposition)
        elif _match_class_path(snapshot, 'gsd.hoomd.Frame'):
            # snapshot is gsd.hoomd.Frame (gsd 2.8+, 3.x)
            snapshot = Snapshot.from_gsd_frame(snapshot,
                                               self._device.communicator)
            self._state = State(self, snapshot, domain_decomposition)
        elif _match_class_path(snapshot, 'gsd.hoomd.Snapshot'):
            # snapshot is gsd.hoomd.Snapshot (gsd 2.x)
            snapshot = Snapshot.from_gsd_snapshot(snapshot,
                                                  self._device.communicator)
            self._state = State(self, snapshot, domain_decomposition)
        else:
            raise TypeError(
                "Snapshot must be a hoomd.Snapshot, gsd.hoomd.Snapshot, "
                "or gsd.hoomd.Frame")

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
        """hoomd.Operations: The operations that apply to the state.

        The operations apply to the state during the simulation run when
        scheduled.

        See Also:
            `run`
        """
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

        Warning:
            The elapsed walltime and timestep are reset at the beginning of each
            call to `run`. Thus, `tps` will provide noisy estimates of
            performance at the start and stable long term averages after
            many timesteps.

        Tip:
            Use the total elapsed wall time and timestep to average the
            timesteps executed per second at desired intervals.

        .. rubric:: Example:

        .. code-block:: python

            logger.add(obj=simulation, quantities=['tps'])
        """
        if self._state is None:
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

        See Also:
            `hoomd.communicator.Communicator.walltime`.

        .. rubric:: Example:

        .. code-block:: python

            logger.add(obj=simulation, quantities=['walltime'])
        """
        if self._state is None:
            return 0.0
        else:
            return self._cpp_sys.walltime

    @log
    def final_timestep(self):
        """float: `run` will end at this timestep.

        `final_timestep` is the timestep on which the currently executing `run`
        will complete.

        .. rubric:: Example:

        .. code-block:: python

            logger.add(obj=simulation, quantities=['final_timestep'])
        """
        if self._state is None:
            return self.timestep
        else:
            return self._cpp_sys.final_timestep

    @log
    def initial_timestep(self):
        """float: `run` started at this timestep.

        `initial_timestep` is the timestep on which the currently executing
        `run` started.

        .. rubric:: Example:

        .. code-block:: python

            logger.add(obj=simulation, quantities=['initial_timestep'])
        """
        if self._state is None:
            return self.timestep
        else:
            return self._cpp_sys.initial_timestep

    @property
    def always_compute_pressure(self):
        """bool: Always compute the virial and pressure (defaults to ``False``).

        By default, HOOMD only computes the virial and pressure on timesteps
        where it is needed (when `hoomd.write.GSD` writes
        log data to a file or when using an NPT integrator). Set
        `always_compute_pressure` to True to make the per particle virial,
        net virial, and system pressure available to query any time by property
        or through the `hoomd.logging.Logger` interface.

        Note:
            Enabling this flag will result in a moderate performance penalty
            when using MD pair potentials.

        .. rubric:: Example:

        .. code-block:: python

            simulation.always_compute_pressure = True
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
               executed before the time step loop.

        Note:
            Initialize the simulation's state before calling `run`.

        `Simulation` applies its `operations` to the
        state during each time step in the order: tuners, updaters, integrator,
        then writers following the logic in this pseudocode::

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

        .. rubric:: Example:

        .. invisible-code-block: python

            simulation = hoomd.util.make_example_simulation()

        .. code-block:: python

            simulation.run(1_000)
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
                             f"{TIMESTEP_MAX - 1}]")

        self._cpp_sys.run(steps_int, write_at_start)

    def __del__(self):
        """Clean up dangling references to simulation."""
        # _operations may not be set, check before unscheduling
        if hasattr(self, "_operations"):
            self._operations._unschedule()


def _match_class_path(obj, *matches):
    return any(cls.__module__ + '.' + cls.__name__ in matches
               for cls in inspect.getmro(type(obj)))
