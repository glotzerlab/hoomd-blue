# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: tommy-waltmann / All Developers are free to add commands for new features

r""" Devices available to run simulations

A device object represents the hardware (whether CPU, GPU, or auto) the simulation will run on. Creating a device
object will automatically add it to the simulation context. A device is chosen by default for the user, but they
can chose a new one before starting the simulation
"""

import os
import time
import socket
import getpass
import hoomd
from hoomd import _hoomd

global mpi_conf, msg


class _device(hoomd.meta._metadata):

    def __init__(self):

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
        if hoomd.context.current.mpi_conf is None:
            raise RuntimeError("Cannot create a device before calling hoomd.context.initialize()")

        # c++ device mirror class instance
        self.cpp_device = None

        # add to simulation context
        hoomd.context.current.device = self

    # \brief Return the network hostname.
    @property
    def hostname(self):
        return socket.gethostname()

    # \brief Return the name of the GPU used in GPU mode.
    @property
    def gpu(self):
        n_gpu = self.cpp_device.getNumActiveGPUs()
        return [self.cpp_device.getGPUName(i) for i in range(n_gpu)]

    # \brief Return the execution mode
    @property
    def mode(self):
        if self.cpp_device.isCUDAEnabled():
            return 'gpu';
        else:
            return 'cpu';

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
            return os.environ['PBS_JOBID'];
        elif 'SLURM_JOB_ID' in os.environ:
            return os.environ['SLURM_JOB_ID'];
        else:
            return '';

    # \brief Return the job name
    @property
    def job_name(self):
        if 'PBS_JOBNAME' in os.environ:
            return os.environ['PBS_JOBNAME'];
        elif 'SLURM_JOB_NAME' in os.environ:
            return os.environ['SLURM_JOB_NAME'];
        else:
            return '';

    # \brief Return the number of CPU threads
    @property
    def num_threads(self):
        if not _hoomd.is_TBB_available():
            msg.warning("HOOMD was compiled without thread support, returning None\n");
            return None
        else:
            return self.cpp_device.getNumThreads();


def _setup_cpp_device(cpp_device, memory_traceback, nthreads):
    """
    Calls some functions on the cpp_device object, to set it up completely
    """

    if _hoomd.is_TBB_available():
        # set the number of TBB threads as necessary
        if nthreads != None:
            cpp_device.setNumThreads(nthreads)

    # set memory traceback
    cpp_device.setMemoryTracing(memory_traceback)


class gpu(_device):

    def __init__(self, memory_traceback=False, min_cpu=None, ignore_display=None, nthreads=None, gpu=None, gpu_error_checking=None):
        """

        :param memory_traceback:
        :param min_cpu: Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)
        :param ignore_display: Attempt to avoid running on the display GPU
        :param nthreads: number of TBB threads
        :param gpu: GPU or comma-separated list of GPUs on which to execute
        :param gpu_error_checking: Enable error checking on the GPU
        """

        _device.__init__(self)

        # convert None options to defaults
        if gpu is None:
            gpu_id = []
        else:
            gpu_id = gpu

        gpu_vec = _hoomd.std_vector_int()
        for gpuid in gpu_id:
            gpu_vec.append(gpuid)

        self.cpp_device = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.GPU,
                                                        gpu_vec,
                                                        min_cpu,
                                                        ignore_display,
                                                        mpi_conf,
                                                        msg)

        # if gpu_error_checking is set, enable it on the GPU
        if gpu_error_checking:
            self.cpp_device.setCUDAErrorChecking(True)

        _setup_cpp_device(self.cpp_device, memory_traceback, nthreads)


class cpu(_device):

    def __init__(self, memory_traceback=False, min_cpu=None, ignore_display=None, nthreads=None):
        """

        :param memory_traceback:
        :param min_cpu: Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)
        :param ignore_display: Attempt to avoid running on the display GPU"
        :param nthreads: number of TBB threads
        """

        _device.__init__(self)

        self.cpp_device = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.CPU,
                                                        _hoomd.std_vector_int(),
                                                        min_cpu,
                                                        ignore_display,
                                                        mpi_conf,
                                                        msg)

        _setup_cpp_device(self.cpp_device, memory_traceback, nthreads)


class auto(_device):

    def __init__(self, memory_traceback=False, min_cpu=None, ignore_display=None, nthreads=None):
        """

        :param memory_traceback:
        :param min_cpu: Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)
        :param ignore_display: Attempt to avoid running on the display GPU"
        :param nthreads: number of TBB threads
        """

        _device.__init__(self)

        self.cpp_device = _hoomd.ExecutionConfiguration(_hoomd.ExecutionConfiguration.executionMode.AUTO,
                                                        _hoomd.std_vector_int(),
                                                        min_cpu,
                                                        ignore_display,
                                                        mpi_conf,
                                                        msg)

        _setup_cpp_device(self.cpp_device, memory_traceback, nthreads)
