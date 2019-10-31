# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: tommy-waltmann / All Developers are free to add commands for new features

r""" Devices available to run simulations

A device object represents the hardware (whether CPU, GPU, or Auto) the simulation will run on. Creating a device
object will automatically add it to the simulation context. A device in mode Auto is chosen by default for the user,
but they can chose a new one before starting the simulation. Devices in mode Auto will choose GPU at runtime if available,
otherwise simulations will run on the CPU.

Note:
    Device objects have the following properties:


Attributes:
    mode (str): gpu or cpu
    comm (:py:mod:`hoomd.comm.Communicator`): communicator object held by this device
    num_threads (int): the number of CPU threads to be used in simulation, settable
    gpu_ids (list(int)): list of names of the gpus, if in gpu mode
    num_ranks (int): the number of ranks
    notice_level (int): minimum level of notice messages to print, settable
    memory_tracback (bool): If true, enable memory allocation tracking (*only for debugging/profiling purposes*)
    gpu_error_checking (bool): Whether or not CUDA error checking is enabled, settable.
    msg_file (str): The name of the file to write messages. Set this property to None to write to stdout/stderr.
"""

import os
import time
import socket
import atexit
import getpass
import hoomd
from hoomd import _hoomd

class _device(hoomd.meta._metadata):

    def __init__(self, communicator, notice_level, msg_file, shared_msg_file):

        # metadata stuff
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = ['gpu_ids', 'mode', 'num_ranks']
        if _hoomd.is_TBB_available():
            self.metadata_fields.append('num_threads')

        # check shared_msg_file
        if shared_msg_file is not None:
            if not _hoomd.is_MPI_available():
                raise RuntimeError("Shared log files are only available in MPI builds.\n")

        # MPI communicator
        if communicator is None:
            self._comm = hoomd.comm.Communicator()
        else:
            self._comm = communicator

        # c++ messenger object
        self.cpp_msg = _create_messenger(self.comm.cpp_mpi_conf, notice_level, msg_file, shared_msg_file)

        # output the version info on initialization
        self.cpp_msg.notice(1, _hoomd.output_version_info())

        # c++ execution configuration mirror class
        self.cpp_exec_conf = None

        # name of the message file
        self._msg_file = msg_file

    @property
    def comm(self):
        """
        Get the MPI Communicator
        """

        return self._comm

    # \brief Return the name of the GPU used in GPU mode.
    @property
    def gpu_ids(self):

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

        return self.comm.num_ranks

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

    @property
    def gpu_error_checking(self):

        if self.mode == 'gpu':
            return self.cpp_exec_conf.isCUDAErrorCheckingEnabled()
        else:
            self.cpp_msg.warning("Attempting to access gpu_error_checking while HOOMD is in CPU mode, returning False.\n")
            return False

    @gpu_error_checking.setter
    def gpu_error_checking(self, new_bool):

        if self.mode == 'gpu':
            self.cpp_exec_conf.setCUDAErrorChecking(new_bool)
        else:
            self.cpp_msg.warning("HOOMD is in CPU mode, ignoring request to set gpu_error_checking.\n")

    @property
    def msg_file(self):

        return self._msg_file

    @msg_file.setter
    def msg_file(self, fname):
        R""" Set the message file.

        Args:
            fname (str): Specifies the name of the file to write. The file will be overwritten.
                         Set to None to direct messages back to stdout/stderr.

        The message file may be changed before or after initialization, and may be changed many times during a job script.
        Changing the message file will only affect messages sent after the change.

        """
        self._msg_file = fname
        if fname is not None:
            self.cpp_msg.openFile(fname)
        else:
            self.cpp_msg.openStd()


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


def _init_nthreads(nthreads):
    """
    initializes the number of threads
    """

    if nthreads is not None:
        self.num_threads = nthreads


class GPU(_device):
    """
    Run simulations on a GPU

    Args:
        gpu_ids (list(int)): GPU or comma-separated list of GPUs on which to execute
        communicator (:py:mod:`hoomd.comm.Communicator`): MPI communicator object. Can be left None if using a
            default MPI communicator
        msg_file (str): Name of file to write messages to
        shared_msg_file (str): (MPI only) Name of shared file to write message to (append partition #)
        notice_level (int): Minimum level of notice messages to print
    """

    def __init__(self, gpu_ids=None, communicator=None, msg_file=None, shared_msg_file=None, notice_level=2):

        _device.__init__(self, communicator, notice_level, msg_file, shared_msg_file)

        # convert None options to defaults
        if gpu_ids is None:
            gpu_id = []
        else:
            gpu_id = gpu_ids

        gpu_vec = _hoomd.std_vector_int()
        for gpuid in gpu_id:
            gpu_vec.append(gpuid)

        self.cpp_exec_conf = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.GPU,
                                                           gpu_vec,
                                                           False,
                                                           False,
                                                           self.comm.cpp_mpi_conf,
                                                           self.cpp_msg)

class CPU(_device):
    """
    Run simulations on a CPU

    Args:
        nthreads (int): number of TBB threads
        communicator (:py:mod:`hoomd.comm.Communicator`): MPI communicator object. Can be left None if using a
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

class Auto(_device):
    """
    Allow simulation hardware to be chosen automatically by HOOMD-blue

    Args:
        nthreads (int): number of TBB threads
        communicator (:py:mod:`hoomd.comm.Communicator`): MPI communicator object. Can be left None if using a
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
