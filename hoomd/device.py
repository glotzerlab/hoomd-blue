# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Devices.

Use a `Device` class to choose which hardware device(s) should execute the
simulation. `Device` also sets where to write log messages and how verbose
the message output should be. Pass a `Device` object to `hoomd.Simulation`
on instantiation to set the options for that simulation.

User scripts may instantiate multiple `Device` objects and use each with a
different `hoomd.Simulation` object. One `Device` object may also be shared
with many `hoomd.Simulation` objects.

Tip:
    Reuse `Device` objects when possible. There is a non-negligible overhead
    to creating each `Device`, especially on the GPU.

See Also:
    `hoomd.Simulation`
"""

import warnings

import contextlib
import hoomd
from hoomd import _hoomd


class NoticeFile:
    """Write a notice message to a file using `hoomd.Device`.

    Args:
        device (`hoomd.Device`): The `Device` object.
    """

    def __init__(self, device):
        self._msg = device._cpp_msg
        self._buff = ""
        self._level = device.notice_level

    def write(self, message, level=None):
        """Writes the notice message to the device.

        Args:
            message (str): Message to write.
            level (int): Message notice level.
        """
        if level is None:
            level = self._level
        self._buff += str(message)
        lines = self._buff.split("\n")
        for line in lines:

            if "\n" == line[-2:]:
                pass
            else:
                line += "\n"

            self._msg.notice(level, line)
        self._buff = ""

    def flush(self):
        """Flush the device."""
        pass


class Device:
    """Base class device object.

    Provides methods and properties common to `CPU` and `GPU`, including those
    that control where status messages are stored (`message_filename`) how many
    status messages HOOMD-blue prints (`notice_level`) and a method for user
    provided status messages (`notice`).

    Warning:
        `Device` cannot be used directly. Instantate a `CPU` or `GPU` object.

    .. rubric:: TBB threads

    Set `num_cpu_threads` to `None` and TBB will auto-select the number of CPU
    threads to execute. If the environment variable ``OMP_NUM_THREADS`` is set,
    HOOMD will use this value. You can also set `num_cpu_threads` explicitly.

    Note:
        At this time **very few** features use TBB for threading. Most users
        should employ MPI for parallel simulations. See `features` for more
        information.
    """

    def __init__(self, communicator, notice_level, message_filename):
        # MPI communicator
        if communicator is None:
            self._comm = hoomd.communicator.Communicator()
        else:
            self._comm = communicator

        # c++ messenger object
        self._cpp_msg = _create_messenger(self.communicator.cpp_mpi_conf,
                                          notice_level, message_filename)

        # c++ execution configuration mirror class
        self._cpp_exec_conf = None

        # name of the message file
        self._message_filename = message_filename

    @property
    def communicator(self):
        """hoomd.communicator.Communicator: The MPI Communicator [read only]."""
        return self._comm

    @property
    def notice_level(self):
        """int: Minimum level of messages to print.

        `notice_level` controls the verbosity of messages printed by HOOMD. The
        default level of 2 shows messages that the developers expect most users
        will want to see. Set the level lower to reduce verbosity or as high as
        10 to get extremely verbose debugging messages.
        """
        return self._cpp_msg.getNoticeLevel()

    @notice_level.setter
    def notice_level(self, notice_level):
        self._cpp_msg.setNoticeLevel(notice_level)

    @property
    def message_filename(self):
        """str: Filename to write messages to.

        By default, HOOMD prints all messages and errors to Python's
        `sys.stdout` and `sys.stderr` (or the system's ``stdout`` and ``stderr``
        when running in an MPI environment).

        Set `message_filename` to a filename to redirect these messages to that
        file.

        Set `message_filename` to `None` to use the system's ``stdout`` and
        ``stderr``.

        Note:
            All MPI ranks within a given partition must open the same file.
            To ensure this, the given file name on rank 0 is broadcast to the
            other ranks. Different partitions may open separate files. For
            example:

            .. code::

                communicator = hoomd.communicator.Communicator(
                    ranks_per_partition=2)
                filename = f'messages.{communicator.partition}'
                device = hoomd.device.GPU(communicator=communicator,
                                          message_filename=filename)
        """
        return self._message_filename

    @property
    def msg_file(self):
        """str: Filename to write messages to.

        .. deprecated:: v3.10.0
           ``msg_file`` will be renamed to ``message_Filename`` in v4.
        """
        warnings.warn(
            "msg_file is deprecated since v3.10.0. Use message_filename.",
            FutureWarning)
        return self.message_filename

    @message_filename.setter
    def message_filename(self, filename):
        self._message_filename = filename
        if filename is not None:
            self._cpp_msg.openFile(filename)
        else:
            self._cpp_msg.openStd()

    @msg_file.setter
    def msg_file(self, filename):
        warnings.warn(
            "msg_file is deprecated since v3.10.0. Use message_filename.",
            FutureWarning)
        self.message_filename = filename

    @property
    def devices(self):
        """list[str]: Descriptions of the active hardware devices."""
        return self._cpp_exec_conf.getActiveDevices()

    @property
    def num_cpu_threads(self):
        """int: Number of TBB threads to use."""
        if not hoomd.version.tbb_enabled:
            return 1
        else:
            return self._cpp_exec_conf.getNumThreads()

    @num_cpu_threads.setter
    def num_cpu_threads(self, num_cpu_threads):
        if not hoomd.version.tbb_enabled:
            self._cpp_msg.warning(
                "HOOMD was compiled without thread support, ignoring request "
                "to set number of threads.\n")
        else:
            self._cpp_exec_conf.setNumThreads(int(num_cpu_threads))

    def notice(self, message, level=1):
        """Write a notice message.

        Args:
            message (str): Message to write.
            level (int): Message notice level.

        Write the given message string to the output defined by
        `message_filename` on MPI rank 0 when `notice_level` >= ``level``.

        Hint:
            Use `notice` instead of `print` to write status messages and your
            scripts will work well in parallel MPI jobs. `notice` writes message
            only on rank 0. Use with a rank-specific `message_filename` to
            troubleshoot issues with specific partitions.
        """
        self._cpp_msg.notice(level, str(message) + "\n")


def _create_messenger(mpi_config, notice_level, message_filename):
    msg = _hoomd.Messenger(mpi_config)

    # try to detect if we're running inside an MPI job
    inside_mpi_job = mpi_config.getNRanksGlobal() > 1

    # only open python stdout/stderr in non-MPI runs
    if not inside_mpi_job:
        msg.openPython()

    if notice_level is not None:
        msg.setNoticeLevel(notice_level)

    if message_filename is not None:
        msg.openFile(message_filename)

    return msg


def _get_message_filename(msg_file, message_file):
    if msg_file is None and message_file is None:
        return None

    if msg_file is not None and message_file is not None:
        raise ValueError("Pass in msg_file or message_file, not both")

    if message_file is not None:
        return message_file

    if msg_file is not None:
        warnings.warn(
            "msg_file is deprecated since v3.10.0. Use message_filename.",
            FutureWarning)
        return msg_file


class GPU(Device):
    """Select a GPU or GPU(s) to execute simulations.

    Args:
        gpu_ids (list[int]): List of GPU ids to use. Set to `None` to let the
            driver auto-select a GPU.

        num_cpu_threads (int): Number of TBB threads. Set to `None` to
            auto-select.

        communicator (hoomd.communicator.Communicator): MPI communicator object.
            When `None`, create a default communicator that uses all MPI ranks.

        msg_file (str): Alias for ``message_filename``.

            .. deprecated:: v3.10.0
               ``msg_file`` will be renamed to ``message_filename`` in v4.

        notice_level (int): Minimum level of messages to print.

        message_filename (str): Filename to write messages to. When `None`, use
            `sys.stdout` and `sys.stderr`. Messages from multiple MPI
            ranks are collected into this file.

    Tip:
        Call `GPU.get_available_devices` to get a human readable list of
        devices. ``gpu_ids = [0]`` will select the first device in this list,
        ``[1]`` will select the second, and so on.

        The ordering of the devices is determined by the GPU driver and runtime.

    .. rubric:: Device auto-selection

    When ``gpu_ids`` is `None`, HOOMD will ask the GPU driver to auto-select a
    GPU. In most cases, this will select device 0. When all devices are set to a
    compute exclusive mode, the driver will choose a free GPU.

    .. rubric:: MPI

    In MPI execution environments, create a `GPU` device on every rank. When
    ``gpu_ids`` is left `None`, HOOMD will attempt to detect the MPI local rank
    environment and choose an appropriate GPU with ``id = local_rank %
    num_capable_gpus``. Set `notice_level` to 3 to see status messages from this
    process. Override this auto-selection by providing appropriate device ids on
    each rank.

    .. rubric:: Multiple GPUs

    Specify a list of GPUs to ``gpu_ids`` to activate a single-process multi-GPU
    code path.

    Note:
        Not all features are optimized to use this code path, and it requires
        that all GPUs support concurrent managed memory access and have high
        bandwidth interconnects.

    """

    def __init__(self,
                 gpu_ids=None,
                 num_cpu_threads=None,
                 communicator=None,
                 msg_file=None,
                 notice_level=2,
                 message_filename=None):

        super().__init__(communicator, notice_level,
                         _get_message_filename(msg_file, message_filename))

        if gpu_ids is None:
            gpu_ids = []

        # convert None options to defaults
        self._cpp_exec_conf = _hoomd.ExecutionConfiguration(
            _hoomd.ExecutionConfiguration.executionMode.GPU, gpu_ids,
            self.communicator.cpp_mpi_conf, self._cpp_msg)

        if num_cpu_threads is not None:
            self.num_cpu_threads = num_cpu_threads

    @property
    def memory_traceback(self):
        """bool: Whether GPU memory tracebacks should be enabled.

        Memory tracebacks are useful for developers when debugging GPU code.

        .. deprecated:: v3.4.0
           `memory_traceback` has no effect.
        """
        warnings.warn("memory_traceback will be removed in hoomd 4.0.",
                      FutureWarning)
        return self._cpp_exec_conf.memoryTracingEnabled()

    @memory_traceback.setter
    def memory_traceback(self, mem_traceback):
        warnings.warn("memory_traceback will be removed in hoomd 4.0.",
                      FutureWarning)
        self._cpp_exec_conf.setMemoryTracing(mem_traceback)

    @property
    def gpu_error_checking(self):
        """bool: Whether to check for GPU error conditions after every call.

        When `False` (the default), error messages from the GPU may not be
        noticed immediately. Set to `True` to increase the accuracy of the GPU
        error messages at the cost of significantly reduced performance.
        """
        return self._cpp_exec_conf.isCUDAErrorCheckingEnabled()

    @gpu_error_checking.setter
    def gpu_error_checking(self, new_bool):
        self._cpp_exec_conf.setCUDAErrorChecking(new_bool)

    @property
    def compute_capability(self):
        """tuple(int, int): Compute capability of the device.

        The tuple includes the major and minor versions of the CUDA compute
        capability: ``(major, minor)``.
        """
        return self._cpp_exec_conf.getComputeCapability(0)

    @staticmethod
    def is_available():
        """Test if the GPU device is available.

        Returns:
            bool: `True` if this build of HOOMD supports GPUs, `False` if not.
        """
        return hoomd.version.gpu_enabled

    @staticmethod
    def get_available_devices():
        """Get the available GPU devices.

        Returns:
            list[str]: Descriptions of the available devices (if any).
        """
        return list(_hoomd.ExecutionConfiguration.getCapableDevices())

    @staticmethod
    def get_unavailable_device_reasons():
        """Get messages describing the reasons why devices are unavailable.

        Returns:
            list[str]: Messages indicating why some devices are unavailable
            (if any).
        """
        return list(_hoomd.ExecutionConfiguration.getScanMessages())

    @contextlib.contextmanager
    def enable_profiling(self):
        """Enable GPU profiling.

        When using GPU profiling tools on HOOMD, select the option to disable
        profiling on start. Initialize and run a simulation long enough that all
        autotuners have completed, then open :py:func:`enable_profiling` as a
        context manager and continue the simulation for a time. Profiling stops
        when the context manager closes.

        Example::

            with device.enable_profiling():
                sim.run(1000)
        """
        try:
            self._cpp_exec_conf.hipProfileStart()
            yield None
        finally:
            self._cpp_exec_conf.hipProfileStop()


class CPU(Device):
    """Select the CPU to execute simulations.

    Args:
        num_cpu_threads (int): Number of TBB threads. Set to `None` to
            auto-select.

        communicator (hoomd.communicator.Communicator): MPI communicator object.
            When `None`, create a default communicator that uses all MPI ranks.

        msg_file (str): Alias for ``message_filename``.

            .. deprecated:: v3.10.0
               ``msg_file`` will be renamed to ``message_filename`` in v4.

        notice_level (int): Minimum level of messages to print.

        message_filename (str): Filename to write messages to. When `None`, use
            `sys.stdout` and `sys.stderr`. Messages from multiple MPI
            ranks are collected into this file.

    .. rubric:: MPI

    In MPI execution environments, create a `CPU` device on every rank.
    """

    def __init__(self,
                 num_cpu_threads=None,
                 communicator=None,
                 msg_file=None,
                 notice_level=2,
                 message_filename=None):

        super().__init__(communicator, notice_level,
                         _get_message_filename(msg_file, message_filename))

        self._cpp_exec_conf = _hoomd.ExecutionConfiguration(
            _hoomd.ExecutionConfiguration.executionMode.CPU, [],
            self.communicator.cpp_mpi_conf, self._cpp_msg)

        if num_cpu_threads is not None:
            self.num_cpu_threads = num_cpu_threads


def auto_select(communicator=None,
                msg_file=None,
                notice_level=2,
                message_filename=None):
    """Automatically select the hardware device.

    Args:

        communicator (hoomd.communicator.Communicator): MPI communicator object.
            When `None`, create a default communicator that uses all MPI ranks.

        msg_file (str): Alias for ``message_filename``.

            .. deprecated:: v3.10.0
               ``msg_file`` will be renamed to ``message_filename`` in v4.

        notice_level (int): Minimum level of messages to print.

        message_filename (str): Filename to write messages to. When `None`, use
            `sys.stdout` and `sys.stderr`. Messages from multiple MPI
            ranks are collected into this file.

    Returns:
        Instance of `GPU` if availabile, otherwise `CPU`.
    """
    # Set class according to C++ object
    if len(GPU.get_available_devices()) > 0:
        return GPU(None, None, communicator,
                   _get_message_filename(msg_file, message_filename),
                   notice_level)
    else:
        return CPU(None, communicator,
                   _get_message_filename(msg_file, message_filename),
                   notice_level)
