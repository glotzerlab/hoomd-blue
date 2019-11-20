# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: csadorf / All Developers are free to add commands for new features

R""" Manage execution contexts.

Every hoomd simulation needs an execution context that describes what hardware it should execute on,
the MPI configuration for the job, etc...
"""

import os
import hoomd
from hoomd import _hoomd
from hoomd import cite


# The following global variables keep track of the walltime and processing time since the import of hoomd
import time
TIME_START = time.time()
CLOCK_START = time.perf_counter()

## Global bibliography
bib = None;

## Global options
options = None;

## Current simulation context
current = None;

class SimulationContext(object):
    R""" Simulation context

    Args:
        device (:py:mod:`hoomd.device`): the device to use for the simulation

    Store all of the context related to a single simulation, including the system state, forces, updaters, integration
    methods, and all other commands specified on this simulation. All such commands in hoomd apply to the currently
    active simulation context. You swap between simulation contexts by using this class as a context manager::


        sim1 = context.SimulationContext();
        sim2 = context.SimulationContext();
        with sim1:
          init.read_xml('init1.xml');
          lj = pair.lj(...)
          ...

        with sim2:
          init.read_xml('init2.xml');
          gauss = pair.gauss(...)
          ...

        # run simulation 1 for a bit
        with sim1:
           run(100)

        # run simulation 2 for a bit
        with sim2:
           run(100)

        # set_current sets the current context without needing to use with
        sim1.set_current()
        run(100)


    If you do not need to maintain multiple contexts, you can call `context.initialize()` to  initialize a new context
    and erase the existing one::

        context.initialize()
        init.read_xml('init1.xml');
        lj = pair.lj(...)
        ...
        run(100);

        context.initialize()
        init.read_xml('init2.xml');
        gauss = pair.gauss(...)
        ...
        run(100)

    Attributes:
        sorter (:py:class:`hoomd.update.sort`): Global particle sorter.
        system_definition (:py:class:`hoomd.data.system_data`): System definition.

    The attributes are global to the context. User scripts may access documented attributes to control settings,
    access particle data, etc... See the linked documentation of each attribute for more details. For example,
    to disable the global sorter::

        c = context.initialize();
        c.sorter.disable();

    """
    def __init__(self, device=None):

        ## Global variable that holds the SystemDefinition shared by all parts of hoomd
        self.system_definition = None;

        ## Global variable that holds the System shared by all parts of hoomd
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

        ## MPCD system
        self.mpcd = None;

        ## Stored reference to the reader that was used to initialize the system
        self.state_reader = None;

        ## Global variable tracking the device used for running the simulation
        ## by default, this is automatically set, unless the user assigns something different
        if device is None:
            self.device = _create_device()
        else:
            self.device = device


    def set_current(self):
        R""" Force this to be the current context
        """
        global current

        current = self;

    def __enter__(self):
        global current

        self.prev = current;
        current = self;
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global current

        current = self.prev;

def initialize(args=None, device=None):
    R""" Initialize the execution context

    Args:
        args (str): Arguments to parse. When *None*, parse the arguments passed on the command line.
        device (:py:mod:`hoomd.device`): device to use for running the simulations

    :py:func:`hoomd.context.initialize()` parses the command line arguments given, sets the options and initializes MPI and GPU execution
    (if any). By default, :py:func:`hoomd.context.initialize()` reads arguments given on the command line. Provide a string to :py:func:`hoomd.context.initialize()`
    to set the launch configuration within the job script.

    :py:func:`hoomd.context.initialize()` can be called more than once in a script. However, the execution parameters are fixed on the first call
    and *args* is ignored. Subsequent calls to :py:func:`hoomd.context.initialize()` create a new :py:class:`SimulationContext` and set it current. This
    behavior is primarily to support use of hoomd in jupyter notebooks, so that a new clean simulation context is
    set when rerunning the notebook within an existing kernel.

    Example::

        from hoomd import *
        context.initialize();
        context.initialize("--mode=gpu --nrank=64");
        context.initialize("--mode=cpu --nthreads=64");

        world = MPI.COMM_WORLD
        comm = world.Split(world.Get_rank(), 0)
        c = comm.Communicator(mpi_comm=comm)
        hoomd.context.initialize(device=device.GPU(communicator=c))

    """
    global options, current

    options = hoomd.option.options();
    hoomd.option._parse_command_line(args);

    current = SimulationContext(device)

    # ensure creation of global bibliography to print HOOMD base citations
    cite._ensure_global_bib()

    return current

# band-aid
def _create_device():

    if options.mode == "gpu":
        dev = hoomd.device.GPU()
        dev.gpu_error_checking = options.gpu_error_checking
    elif options.mode == "cpu":
        dev = hoomd.device.CPU()
    else:
        dev = hoomd.device.Auto()

    return dev

## \internal
# \brief Throw an error if the context is not initialized
def _verify_init():
    global current

    if current is None:
        raise RuntimeError("Call context.initialize() before any method")

## \internal
# \brief Gather context about HOOMD
class HOOMDContext(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the context object
    def __init__(self):
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hoomd_version', 'hoomd_git_sha1', 'hoomd_git_refspec',
            'hoomd_compile_flags', 'cuda_version', 'compiler_version',
            ]

    # \brief Return the hoomd version.
    @property
    def hoomd_version(self):
        return _hoomd.__version__

    # \brief Return the hoomd git hash
    @property
    def hoomd_git_sha1(self):
        return _hoomd.__git_sha1__

    # \brief Return the hoomd git refspec
    @property
    def hoomd_git_refspec(self):
        return _hoomd.__git_refspec__

    # \brief Return the hoomd compile flags
    @property
    def hoomd_compile_flags(self):
        return _hoomd.hoomd_compile_flags();

    # \brief Return the cuda version
    @property
    def cuda_version(self):
        return _hoomd.__cuda_version__

    # \brief Return the compiler version
    @property
    def compiler_version(self):
        return _hoomd.__compiler_version__
