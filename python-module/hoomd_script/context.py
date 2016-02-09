# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
# the University of Michigan All rights reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: csadorf / All Developers are free to add commands for new features

## \package hoomd_script.context
# \brief Gather information about the execution context
#
# As much data from the environment is gathered as possible.

import os
import hoomd;
import hoomd_script
import socket
import getpass

# The following global variables keep track of the walltime and processing time since the import of hoomd_script
import time
TIME_START = time.time()
CLOCK_START = time.clock()

## Global Messenger
# \note This is initialized to a default messenger on load so that python code may have a unified path for sending
# messages
msg = hoomd.Messenger();

## Global bibliography
bib = None;

## Global options
options = None;

## Global variable that holds the execution configuration for reference by the python API
exec_conf = None;

## Current simulation context
current = None;

## Simulation context
#
# Store all of the context related to a single simulation, including the system state, forces, updaters, integration
# methods, and all other commands specified on this simulation. All such commands in hoomd apply to the currently
# active simulation context. You swap between simulation contexts by using this class as a context manager:
#
# ```
# sim1 = context.SimulationContext();
# sim2 = context.SimulationContext();
# with sim1:
#   init.read_xml('init1.xml');
#   lj = pair.lj(...)
#   ...
#
# with sim2:
#   init.read_xml('init2.xml');
#   gauss = pair.gauss(...)
#   ...
#
# # run simulation 1 for a bit
# with sim1:
#    run(100)
#
# # run simulation 2 for a bit
# with sim2:
#    run(100)
# ```
#
# If you do not need to maintain multiple contexts, you can call `context.initialize()` to  initialize a new context
# and erase the existing one.
#
# ```
# context.initialize()
# init.read_xml('init1.xml');
# lj = pair.lj(...)
# ...
# run(100);
#
# context.initialize()
# init.read_xml('init2.xml');
# gauss = pair.gauss(...)
# ...
# run(100)
# ```
class SimulationContext(object):
    def __init__(self):
        ## Global variable that holds the SystemDefinition shared by all parts of hoomd_script
        self.system_definition = None;

        ## Global variable that holds the System shared by all parts of hoomd_script
        self.system = None;

        ## Global variable that holds the balanced domain decomposition in MPI runs if it is requested
        self.decomposition = None

        ## Global variable that holds the sorter
        self.sorter = None;

        ## Global variable that tracks the all of the force computes specified in the script so far
        self.forces = [];

        ## Global variable that tracks the all of the constraint force computes specified in the script so far
        self.constraint_forces = [];

        ## Global variable that tracks all the integration methods that have been specified in the script so far
        self.integration_methods = [];

        ## Global variable tracking the last _integrator set
        self.integrator = None;

        ## Global variable tracking the system's neighborlist
        self.neighbor_list = None;

        ## Global variable tracking all neighbor lists that have been created
        self.neighbor_lists = []

        ## Global variable tracking all the loggers that have been created
        self.loggers = [];

        ## Global variable tracking all the analyzers that have been created
        self.analyzers = [];

        ## Global variable tracking all the updaters that have been created
        self.updaters = [];

        ## Global variable tracking all the compute thermos that have been created
        self.thermos = [];

        ## Cached all group
        self.group_all = None;

    def __enter__(self):
        global current

        current = self;

## Initialize the execution context
# \param args Arguments to parse. When \a None, parse the arguments passed on the command line.
#
# initialize() parses the command line arguments given, sets the options and initializes MPI and GPU execution
# (if any). By default, initialize() reads arguments given on the command line. Provide a string to initialize()
# to set the launch configuration within the job script.
#
# initialize() should be called immediately after `from hoomd_script import *`.
#
# **Example:**
# \code
# from hoomd_script import *
# context.initialize();
# context.initialize("--mode=gpu --nrank=64");
# \endcode
#
def initialize(args=None):
    global exec_conf, msg, options, current

    if exec_conf is not None:
        msg.error("Cannot change execution mode after initialization\n");
        raise RuntimeError('Error setting option');

    options = hoomd_script.option.options();
    hoomd_script.option._parse_command_line(args);

    _create_exec_conf();

    current = SimulationContext();

## Get the current processor name
#
# platform.node() can spawn forked processes in some version of MPI.
# This avoids that problem by using MPI information about the hostname directly
# when it is available. MPI is initialized on module load if it is available,
# so this data is accessible immediately.
#
# \returns String name for the current processor
# \internal
def _get_proc_name():
    if hoomd.is_MPI_available():
        return hoomd.get_mpi_proc_name()
    else:
        return platform.node()

## Initializes the execution configuration
#
# \internal
def _create_exec_conf():
    global exec_conf, options, msg

    # use a cached execution configuration if available
    if exec_conf is not None:
        return exec_conf

    mpi_available = hoomd.is_MPI_available();

    # error out on nyx/flux if the auto mode is set
    if options.mode == 'auto':
        host = _get_proc_name()
        if "flux" in host or "nyx" in host:
            msg.error("--mode=gpu or --mode=cpu must be specified on nyx/flux\n");
            raise RuntimeError("Error initializing");
        exec_mode = hoomd.ExecutionConfiguration.executionMode.AUTO;
    elif options.mode == "cpu":
        exec_mode = hoomd.ExecutionConfiguration.executionMode.CPU;
    elif options.mode == "gpu":
        exec_mode = hoomd.ExecutionConfiguration.executionMode.GPU;
    else:
        raise RuntimeError("Invalid mode");

    # convert None options to defaults
    if options.gpu is None:
        gpu_id = -1;
    else:
        gpu_id = int(options.gpu);

    if options.nrank is None:
        nrank = 0;
    else:
        nrank = int(options.nrank);

    # create the specified configuration
    exec_conf = hoomd.ExecutionConfiguration(exec_mode, gpu_id, options.min_cpu, options.ignore_display, msg, nrank);

    # if gpu_error_checking is set, enable it on the GPU
    if options.gpu_error_checking:
       exec_conf.setCUDAErrorChecking(True);

    exec_conf = exec_conf;

    return exec_conf;

## \internal
# \brief Throw an error if the context is not initialized
def _verify_init():
    global exec_conf, msg, current

    if exec_conf is None:
        msg.error("call context.initialize() before any other method in hoomd.")
        raise RuntimeError("hoomd execution context is not available")

## \internal
# \brief Gather context from the environment
class ExecutionContext(hoomd_script.meta._metadata):
    ## \internal
    # \brief Constructs the context object
    def __init__(self):
        hoomd_script.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hostname', 'gpu', 'mode', 'num_ranks',
            'username', 'wallclocktime', 'cputime',
            'job_id', 'job_name'
            ]

    ## \internal
    # \brief Return the execution configuration if initialized or raise exception.
    def _get_exec_conf(self):
        global exec_conf
        if exec_conf is None:
            raise RuntimeError("Not initialized.")
        else:
            return exec_conf

    # \brief Return the network hostname.
    @property
    def hostname(self):
        return socket.gethostname()

    # \brief Return the name of the GPU used in GPU mode.
    @property
    def gpu(self):
        return self._get_exec_conf().getGPUName()

    # \brief Return the execution mode
    @property
    def mode(self):
        if self._get_exec_conf().isCUDAEnabled():
            return 'gpu';
        else:
            return 'cpu';

    # \brief Return the number of ranks.
    @property
    def num_ranks(self):
        return hoomd_script.comm.get_num_ranks()

    # \brief Return the username.
    @property
    def username(self):
        return getpass.getuser()

    # \brief Return the wallclock time since the import of hoomd_script
    @property
    def wallclocktime(self):
        return time.time() - TIME_START

    # \brief Return the CPU clock time since the import of hoomd_script
    @property
    def cputime(self):
        return time.clock() - CLOCK_START

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


## \internal
# \brief Gather context about HOOMD
class HOOMDContext(hoomd_script.meta._metadata):
    ## \internal
    # \brief Constructs the context object
    def __init__(self):
        hoomd_script.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hoomd_version', 'hoomd_git_sha1', 'hoomd_git_refspec',
            'hoomd_compile_flags', 'cuda_version', 'compiler_version',
            ]

    # \brief Return the hoomd version.
    @property
    def hoomd_version(self):
        return hoomd.__version__

    # \brief Return the hoomd git hash
    @property
    def hoomd_git_sha1(self):
        return hoomd.__git_sha1__

    # \brief Return the hoomd git refspec
    @property
    def hoomd_git_refspec(self):
        return hoomd.__git_refspec__

    # \brief Return the hoomd compile flags
    @property
    def hoomd_compile_flags(self):
        return hoomd.hoomd_compile_flags();

    # \brief Return the cuda version
    @property
    def cuda_version(self):
        return hoomd.__cuda_version__

    # \brief Return the compiler version
    @property
    def compiler_version(self):
        return hoomd.__compiler_version__
