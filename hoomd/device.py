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
global mpi_conf, msg, options

class _device(hoomd.meta._metadata):

    def __init__(self, memory_traceback=False):
        global mpi_conf, msg, options

        # metadata stuff
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hostname', 'gpu', 'mode', 'num_ranks',
            'username', 'wallclocktime', 'cputime',
            'job_id', 'job_name'
        ]
        if _hoomd.is_TBB_available():
            self.metadata_fields.append('num_threads')

        # TODO make sure context is initialized

        # c++ device mirror class instance
        self.cpp_device = _create_exec_conf(mpi_conf, msg, options)
        self.cpp_device.setMemoryTracing(memory_traceback)

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

## Initializes the execution configuration
#
# \internal
def _create_exec_conf(mpi_conf, msg, options):

    if options.mode == 'auto':
        exec_mode = _hoomd.ExecutionConfiguration.executionMode.AUTO
    elif options.mode == "cpu":
        exec_mode = _hoomd.ExecutionConfiguration.executionMode.CPU
    elif options.mode == "gpu":
        exec_mode = _hoomd.ExecutionConfiguration.executionMode.GPU
    else:
        raise RuntimeError("Invalid mode")

    # convert None options to defaults
    if options.gpu is None:
        gpu_id = []
    else:
        gpu_id = options.gpu

    gpu_vec = _hoomd.std_vector_int()
    for gpuid in gpu_id:
        gpu_vec.append(gpuid)

    # create the specified configuration
    exec_conf = _hoomd.ExecutionConfiguration(exec_mode, gpu_vec, options.min_cpu, options.ignore_display, mpi_conf, msg)

    # if gpu_error_checking is set, enable it on the GPU
    if options.gpu_error_checking:
       exec_conf.setCUDAErrorChecking(True)

    if _hoomd.is_TBB_available():
        # set the number of TBB threads as necessary
        if options.nthreads != None:
            exec_conf.setNumThreads(options.nthreads)

    return exec_conf

class gpu(_device):

    def __init__(self, memory_traceback=False):

        _device.__init__(self, memory_traceback=memory_traceback)


class cpu(_device):

    def __init__(self, memory_traceback=False):

        _device.__init__(self, memory_traceback=memory_traceback)


class auto(_device):

    def __init__(self, memory_traceback=False):

        _device.__init__(self, memory_traceback=memory_traceback)
