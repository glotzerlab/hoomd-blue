# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: tommy-waltmann / All Developers are free to add commands for new features

r""" Devices available to run simulations

A device object represents the hardware (whether CPU, GPU, or AUTO) the simulation will run on. Creating a device
object will automatically add it to the simulation context. A device in mode AUTO is chosen by default for the user,
but they can chose a new one before starting the simulation. Devices in mode AUTO will choose GPU at runtime if available,
otherwise simulations will run on the CPU

Note:
    Device objects have the following properties:


Attributes:
    mode (str): gpu or cpu
    num_threads (int): the number of CPU threads to be used in simulation, settable
    gpu (list(int)): list of names of the gpus, if in gpu mode
    num_ranks (int): the number of ranks
    username (str): the username
    wallclocktime (float): elapsed time since the hoomd script first started execution
    cputime (float): CPU clock time elapsed since the hoomd script first started execution
    job_id (int): the id of the job
    job_name (str): the job name
    notice_level (int): minimum level of notice messages to print, settable
    hostname (str): the network hostname
"""

import os
import time
import socket
import getpass
import hoomd
from hoomd import _hoomd


class _device(hoomd.meta._metadata):

    def __init__(self, mpi_comm, nrank, notice_level, msg_file, shared_msg_file):

        # metadata stuff
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hostname', 'gpu', 'mode', 'num_ranks',
            'username', 'wallclocktime', 'cputime',
            'job_id', 'job_name'
        ]
        if _hoomd.is_TBB_available():
            self.metadata_fields.append('num_threads')

        # make sure context is initialized
        hoomd.context._verify_init()

        # check nrank
        if nrank is not None:
            if not _hoomd.is_MPI_available():
                raise RuntimeError("The nrank option is only available in MPI builds.\n")

        # check shared_msg_file
        if shared_msg_file is not None:
            if not _hoomd.is_MPI_available():
                raise RuntimeError("Shared log files are only available in MPI builds.\n")

        # c++ mpi configuration mirror class instance
        self.cpp_mpi_conf = _create_mpi_conf(mpi_comm, nrank)

        # c++ messenger object
        self.cpp_msg = _create_messenger(self.cpp_mpi_conf, notice_level, msg_file, shared_msg_file)

        # output the version info on initialization
        self.cpp_msg.notice(1, _hoomd.output_version_info())

        # c++ execution configuration mirror class
        self.cpp_exec_conf = None

        # add to simulation context
        hoomd.context.current.device = self

    # \brief Return the network hostname.
    @property
    def hostname(self):
        return socket.gethostname()

    # \brief Return the name of the GPU used in GPU mode.
    @property
    def gpu(self):
        n_gpu = self.cpp_exec_conf.getNumActiveGPUs()
        return [self.cpp_exec_conf.getGPUName(i) for i in range(n_gpu)]

    # \brief Return the execution mode
    @property
    def mode(self):
        if self.cpp_exec_conf.isCUDAEnabled():
            return 'gpu'
        else:
            return 'cpu'

    # \brief Return the number of ranks.
    @property
    def num_ranks(self):
        return hoomd.comm.get_num_ranks()

    # \brief Return the username.
    @property
    def username(self):
        return getpass.getuser()

    # \brief Return the wallclock time since the import of hoomd
    @property
    def wallclocktime(self):
        return time.time() - hoomd.context.TIME_START

    # \brief Return the CPU clock time since the import of hoomd
    @property
    def cputime(self):
        return time.clock() - hoomd.context.CLOCK_START

    # \brief Return the job id
    @property
    def job_id(self):
        if 'PBS_JOBID' in os.environ:
            return os.environ['PBS_JOBID']
        elif 'SLURM_JOB_ID' in os.environ:
            return os.environ['SLURM_JOB_ID']
        else:
            return ''

    # \brief Return the job name
    @property
    def job_name(self):
        if 'PBS_JOBNAME' in os.environ:
            return os.environ['PBS_JOBNAME']
        elif 'SLURM_JOB_NAME' in os.environ:
            return os.environ['SLURM_JOB_NAME']
        else:
            return ''

    # \brief Return the number of CPU threads
    @property
    def num_threads(self):
        if not _hoomd.is_TBB_available():
            self.cpp_msg.warning("HOOMD was compiled without thread support, returning None\n")
            return None
        else:
            return self.cpp_exec_conf.getNumThreads()

    @num_threads.setter
    def num_threads(self, num_threads):
        R""" Set the number of CPU (TBB) threads HOOMD uses

        Args:
            num_threads (int): The number of threads

        Note:
            Overrides ``--nthreads`` on the command line.

        """

        if not _hoomd.is_TBB_available():
            self.cpp_msg.warning("HOOMD was compiled without thread support, ignoring request to set number of threads.\n")
        else:
            self.cpp_exec_conf.setNumThreads(int(num_threads))

    @property
    def notice_level(self):
        return self.cpp_msg.getNoticeLevel()

    @notice_level.setter
    def notice_level(self, notice_level):
        R""" Set the notice level.

        Args:
            notice_level (int). The maximum notice level to print.

        The notice level may be changed before or after initialization, and may be changed
        many times during a job script.

        """

        self.cpp_msg.setNoticeLevel(notice_level)

    def set_msg_file(self, fname):
        R""" Set the message file.

        Args:
            fname (str): Specifies the name of the file to write. The file will be overwritten.
                         Set to None to direct messages back to stdout/stderr.

        The message file may be changed before or after initialization, and may be changed many times during a job script.
        Changing the message file will only affect messages sent after the change.

        """

        if fname is not None:
            self.cpp_msg.openFile(fname)
        else:
            self.cpp_msg.openStd()


def _setup_cpp_exec_conf(cpp_exec_conf, memory_traceback, nthreads):
    """
    Calls some functions on the cpp_exec_conf object, to set it up completely
    """

    if _hoomd.is_TBB_available():
        # set the number of TBB threads as necessary
        if nthreads != None:
            cpp_exec_conf.setNumThreads(nthreads)

    # set memory traceback
    cpp_exec_conf.setMemoryTracing(memory_traceback)


## Initializes the MPI configuration
# \internal
def _create_mpi_conf(mpi_comm, nrank):

    mpi_available = _hoomd.is_MPI_available();

    mpi_conf = None

    # create the specified configuration
    if mpi_comm is None:
        mpi_conf = _hoomd.MPIConfiguration();
    else:
        if not mpi_available:
            raise RuntimeError("mpi_comm is not supported in serial builds");

        handled = False;

        # pass in pointer to MPI_Comm object provided by mpi4py
        try:
            import mpi4py
            if isinstance(mpi_comm, mpi4py.MPI.Comm):
                addr = mpi4py.MPI._addressof(mpi_comm);
                mpi_conf = _hoomd.MPIConfiguration._make_mpi_conf_mpi_comm(addr);
                handled = True
        except ImportError:
            # silently ignore when mpi4py is missing
            pass

        # undocumented case: handle plain integers as pointers to MPI_Comm objects
        if not handled and isinstance(mpi_comm, int):
            mpi_conf = _hoomd.MPIConfiguration._make_mpi_conf_mpi_comm(mpi_comm);
            handled = True

        if not handled:
            raise RuntimeError("Invalid mpi_comm object: {}".format(mpi_comm));

    if nrank is not None:
        # check validity
        if (mpi_conf.getNRanksGlobal() % nrank):
            raise RuntimeError('Total number of ranks is not a multiple of --nrank');

        # split the communicator into partitions
        mpi_conf.splitPartitions(nrank)

    return mpi_conf


## Initializes the Messenger
# \internal
def _create_messenger(mpi_config, notice_level, msg_file, shared_msg_file):

    msg = _hoomd.Messenger(mpi_config)

    # try to detect if we're running inside an MPI job
    inside_mpi_job = mpi_config.getNRanksGlobal() > 1
    if ('OMPI_COMM_WORLD_RANK' in os.environ or
        'MV2_COMM_WORLD_LOCAL_RANK' in os.environ or
        'PMI_RANK' in os.environ or
        'ALPS_APP_PE' in os.environ):
        inside_mpi_job = True

    # only open python stdout/stderr in non-MPI runs
    if not inside_mpi_job:
        msg.openPython();

    if notice_level is not None:
        msg.setNoticeLevel(notice_level);

    if msg_file is not None:
        msg.openFile(msg_file);

    if shared_msg_file is not None:
        if not _hoomd.is_MPI_available():
            msg.error("Shared log files are only available in MPI builds.\n");
            raise RuntimeError('Error setting option');
        msg.setSharedFile(shared_msg_file);

    return msg


def _check_exec_conf_args(nthreads):

    # check nthreads
    if nthreads is not None:
        if not _hoomd.is_TBB_available():
            raise RuntimeError("The nthreads option is only available in TBB-enabled builds.\n");


class gpu(_device):
    """
    Run simulations on a GPU

    Args:
        memory_tracback (bool): If true, enable memory allocation tracking (*only for debugging/profiling purposes*)
        min_cpu (bool): Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)
        ignore_display (bool): Attempt to avoid running on the display GPU
        nthreads (int): number of TBB threads
        gpu (list(int)): GPU or comma-separated list of GPUs on which to execute
        gpu_error_checking (bool): Enable error checking on the GPU
        mpi_comm (:py:mod:`mpi4py.MPI.Comm`): Accepts an mpi4py communicator. Use this argument to perform many independent hoomd simulations
                where you communicate between those simulations using your own mpi4py code.
        nrank (int): (MPI) Number of ranks to include in a partition
        notice_level (int): Minimum level of notice messages to print
        msg_file (str): Name of file to write messages to
        shared_msg_file (str): (MPI only) Name of shared file to write message to (append partition #)
    """

    def __init__(self, memory_traceback=False, min_cpu=False, ignore_display=False, nthreads=None, gpu=None,
                 gpu_error_checking=False, mpi_comm=None, nrank=None, notice_level=2, msg_file=None, shared_msg_file=None):

        _device.__init__(self, mpi_comm, nrank, notice_level, msg_file, shared_msg_file)

        # check args
        _check_exec_conf_args(nthreads)

        # convert None options to defaults
        if gpu is None:
            gpu_id = []
        else:
            gpu_id = gpu

        gpu_vec = _hoomd.std_vector_int()
        for gpuid in gpu_id:
            gpu_vec.append(gpuid)

        self.cpp_exec_conf = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.GPU,
                                                        gpu_vec,
                                                        min_cpu,
                                                        ignore_display,
                                                        self.cpp_mpi_conf,
                                                        self.cpp_msg)

        # if gpu_error_checking is set, enable it on the GPU
        if gpu_error_checking:
            self.cpp_exec_conf.setCUDAErrorChecking(True)

        _setup_cpp_exec_conf(self.cpp_exec_conf, memory_traceback, nthreads)


class cpu(_device):
    """
    Run simulations on a CPU

    Args:
        memory_tracback (bool): If true, enable memory allocation tracking (*only for debugging/profiling purposes*)
        min_cpu (bool): Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)
        ignore_display (bool): Attempt to avoid running on the display GPU
        nthreads (int): number of TBB threads
        mpi_comm (:py:mod:`mpi4py.MPI.Comm`): Accepts an mpi4py communicator. Use this argument to perform many independent hoomd simulations
                where you communicate between those simulations using your own mpi4py code.
        nrank (int): (MPI) Number of ranks to include in a partition
        notice_level (int): Minimum level of notice messages to print
        msg_file (str): Name of file to write messages to
        shared_msg_file (str): (MPI only) Name of shared file to write message to (append partition #)
    """

    def __init__(self, memory_traceback=False, min_cpu=False, ignore_display=False, nthreads=None, mpi_comm=None,
                 nrank=None, notice_level=2, msg_file=None, shared_msg_file=None):

        _device.__init__(self, mpi_comm, nrank, notice_level, msg_file, shared_msg_file)

        _check_exec_conf_args(nthreads)

        self.cpp_exec_conf = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.CPU,
                                                        _hoomd.std_vector_int(),
                                                        min_cpu,
                                                        ignore_display,
                                                        self.cpp_mpi_conf,
                                                        self.cpp_msg)

        _setup_cpp_exec_conf(self.cpp_exec_conf, memory_traceback, nthreads)


class auto(_device):
    """
    Allow simulation hardware to be chosen automatically by HOOMD-blue

    Args:
        memory_tracback (bool): If true, enable memory allocation tracking (*only for debugging/profiling purposes*)
        min_cpu (bool): Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)
        ignore_display (bool): Attempt to avoid running on the display GPU
        nthreads (int): number of TBB threads
        mpi_comm (:py:mod:`mpi4py.MPI.Comm`): Accepts an mpi4py communicator. Use this argument to perform many independent hoomd simulations
                where you communicate between those simulations using your own mpi4py code.
        nrank (int): (MPI) Number of ranks to include in a partition
        notice_level (int): Minimum level of notice messages to print
        msg_file (str): Name of file to write messages to
        shared_msg_file (str): (MPI only) Name of shared file to write message to (append partition #)
    """
    def __init__(self, memory_traceback=False, min_cpu=False, ignore_display=False, nthreads=None, mpi_comm=None,
                 nrank=None, notice_level=2, msg_file=None, shared_msg_file=None):

        _device.__init__(self, mpi_comm, nrank, notice_level, msg_file, shared_msg_file)

        _check_exec_conf_args(nthreads)

        self.cpp_exec_conf = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.AUTO,
                                                        _hoomd.std_vector_int(),
                                                        min_cpu,
                                                        ignore_display,
                                                        self.cpp_mpi_conf,
                                                        self.cpp_msg)

        _setup_cpp_exec_conf(self.cpp_exec_conf, memory_traceback, nthreads)
