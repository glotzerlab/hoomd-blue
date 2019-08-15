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
    memory_tracback (bool): If true, enable memory allocation tracking (*only for debugging/profiling purposes*)
"""

import os
import time
import socket
import atexit
import getpass
import hoomd
from hoomd import _hoomd

# Global list of messengers for proper messenger destruction
_cpp_msgs = []

# this function destroys all the messenger objects that have been created throughout the execution of a script
@atexit.register
def _destroy_messengers():
    global _cpp_msgs

    for msg in _cpp_msgs:
        msg.close()


class _device(hoomd.meta._metadata):

    def __init__(self, communicator, notice_level, msg_file, shared_msg_file):

        # metadata stuff
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hostname', 'gpu', 'mode', 'num_ranks',
            'username', 'wallclocktime', 'cputime',
            'job_id', 'job_name'
        ]
        if _hoomd.is_TBB_available():
            self.metadata_fields.append('num_threads')

        # check shared_msg_file
        if shared_msg_file is not None:
            if not _hoomd.is_MPI_available():
                raise RuntimeError("Shared log files are only available in MPI builds.\n")

        # MPI communicator
        if communicator is None:
            self.comm = hoomd.comm.communicator()
        else:
            self.comm = communicator

        # c++ messenger object
        self.cpp_msg = _create_messenger(self.comm.cpp_mpi_conf, notice_level, msg_file, shared_msg_file)

        # output the version info on initialization
        self.cpp_msg.notice(1, _hoomd.output_version_info())

        # c++ execution configuration mirror class
        self.cpp_exec_conf = None

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
        return self.comm.get_num_ranks()

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
            return 1
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

    @property
    def memory_traceback():
        return self.cpp_exec_conf.getMemoryTracer() is not None
        
    @memory_traceback.setter
    def memory_traceback(self, mem_traceback):
        self.cpp_exec_conf.setMemoryTracing(mem_traceback)

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


## Initializes the Messenger
# \internal
def _create_messenger(mpi_config, notice_level, msg_file, shared_msg_file):
    global _cpp_msgs

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

    # add this messenger to the global list of messengers
    _cpp_msgs.append(msg)

    return msg


def _init_nthreads(nthreads):
    """
    initializes the number of threads
    """
    
    if nthreads is not None:
        self.num_threads = nthreads


class gpu(_device):
    """
    Run simulations on a GPU

    Args:
        gpu (list(int)): GPU or comma-separated list of GPUs on which to execute
        communicator (:py:mod:`hoomd.comm.communicator`): MPI communicator object. Can be left None if using a
            default MPI communicator
        msg_file (str): Name of file to write messages to
        shared_msg_file (str): (MPI only) Name of shared file to write message to (append partition #)
        notice_level (int): Minimum level of notice messages to print
    """

    def __init__(self, gpu=None, communicator=None, msg_file=None, shared_msg_file=None, notice_level=2):

        _device.__init__(self, communicator, notice_level, msg_file, shared_msg_file)

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
                                                           False, 
                                                           False, 
                                                           self.comm.cpp_mpi_conf,
                                                           self.cpp_msg)

    @property
    def gpu_error_checking(self):
        """
        (bool) Whether or not CUDA error checking is enabled, settable.
        """
        
        return self.cpp_exec_conf.isCUDAErrorCheckingEnabled()
    
    @gpu_error_checking.setter
    def gpu_error_checking(self, new_bool):
        self.cpp_exec_conf.setCUDAErrorChecking(new_bool)

class cpu(_device):
    """
    Run simulations on a CPU

    Args:
        nthreads (int): number of TBB threads
        communicator (:py:mod:`hoomd.comm.communicator`): MPI communicator object. Can be left None if using a
            default MPI communicator
        msg_file (str): Name of file to write messages to
        shared_msg_file (str): (MPI only) Name of shared file to write message to (append partition #)
        notice_level (int): Minimum level of notice messages to print
    """

    def __init__(self, nthreads=None, communicator=None, msg_file=None, shared_msg_file=None, notice_level=2):

        _device.__init__(self, communicator, notice_level, msg_file, shared_msg_file)

        _init_nthreads(nthreads)

        self.cpp_exec_conf = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.CPU,
                                                           _hoomd.std_vector_int(), 
                                                           False, 
                                                           False, 
                                                           self.comm.cpp_mpi_conf,
                                                           self.cpp_msg)


class auto(_device):
    """
    Allow simulation hardware to be chosen automatically by HOOMD-blue

    Args:
        nthreads (int): number of TBB threads
        communicator (:py:mod:`hoomd.comm.communicator`): MPI communicator object. Can be left None if using a
            default MPI communicator
        msg_file (str): Name of file to write messages to
        shared_msg_file (str): (MPI only) Name of shared file to write message to (append partition #)
        notice_level (int): Minimum level of notice messages to print
    """

    def __init__(self, nthreads=None, communicator=None, msg_file=None, shared_msg_file=None, notice_level=2):

        _device.__init__(self, communicator, notice_level, msg_file, shared_msg_file)

        _init_nthreads(nthreads)

        self.cpp_exec_conf = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.AUTO,
                                                           _hoomd.std_vector_int(), 
                                                           False, 
                                                           False, 
                                                           self.comm.cpp_mpi_conf, 
                                                           self.cpp_msg)
