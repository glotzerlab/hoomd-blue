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

# Maintainer: joaander
import sys;
import ctypes;
import os;

# need to import HOOMD with RTLD_GLOBAL in python sitedir builds
if not ('NOT_HOOMD_PYTHON_SITEDIR' in os.environ):
    flags = sys.getdlopenflags();
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL);

import hoomd;

if not ('NOT_HOOMD_PYTHON_SITEDIR' in os.environ):
    sys.setdlopenflags(flags);

from hoomd_script import util as _util;
from hoomd_script import init;
from hoomd_script import analyze;
from hoomd_script import bond;
from hoomd_script import benchmark;
from hoomd_script import angle;
from hoomd_script import dihedral;
from hoomd_script import improper;
from hoomd_script import dump;
from hoomd_script import force;
from hoomd_script import external;
from hoomd_script import constrain;
from hoomd_script import globals;
from hoomd_script import group;
from hoomd_script import integrate;
from hoomd_script import option;
from hoomd_script import nlist;
from hoomd_script import pair;
from hoomd_script import sorter;
from hoomd_script import update;
from hoomd_script import wall;
from hoomd_script import variant;
from hoomd_script import tune;
from hoomd_script import hoomd;
from hoomd_script import compute;
from hoomd_script import charge;
from hoomd_script import comm;
from hoomd_script import meta;
from hoomd_script import cite;
from hoomd_script import data;
from hoomd_script import context;

from hoomd import WalltimeLimitReached;

## \package hoomd_script
# \brief Base module for the user-level scripting API
#
# hoomd_script provides a very high level user interface for executing
# simulations using HOOMD. This python module is designed to be imported
# into python with "from hoomd_script import *"

# output the version info on import
context.msg.notice(1, hoomd.output_version_info())

# ensure creation of global bibliography to print HOOMD base citations
cite._ensure_global_bib()

_default_excepthook = sys.excepthook;

## \internal
# \brief Override pythons except hook to abort MPI runs
def _hoomd_sys_excepthook(type, value, traceback):
    _default_excepthook(type, value, traceback);
    sys.stderr.flush();
    if context.exec_conf is not None:
        hoomd.abort_mpi(context.exec_conf);

# install the hoomd excepthook to abort MPI runs if there are uncaught exceptions
sys.excepthook = _hoomd_sys_excepthook;

## \internal
# \brief Major version of hoomd_script
_version_major = 1;

## \internal
# \brief Minor version of hoomd_script
_version_minor = 0;

## \brief Get the version information of hoomd_script
# \returns a tuple (major, minor)
#
# This version is the version number of \b hoomd_script , not HOOMD as a whole. It is intended for use by
# third party API plugins that interface with hoomd_script. When new features are added (i.e. a new command
# or a new option to an existing command), the minor version will be incremented. When major changes are implemented
# or changes that break backwards compatibility are made, then the major version is incremented and the minor reset
# to 0. Only one such increment of either type will occur per each tagged release of HOOMD.
def get_hoomd_script_version():
    return (_version_major, _version_minor)

## \brief Runs the simulation for a given number of time steps
#
# \param tsteps Number of time steps to advance the simulation
# \param profile Set to True to enable detailed profiling
# \param limit_hours  (if set) Limit the run to a given number of hours.
# \param limit_multiple When stopping the run() due to walltime limits, only stop when the time step is a multiple of
#                       \a limit_multiple .
# \param callback     (if set) Sets a Python function to be called regularly during a run.
# \param callback_period Sets the period, in time steps, between calls made to \a callback
# \param quiet Set to True to eliminate the status information printed to the screen by the run
#
# \b Examples:
# \code
# run(1000)
# run(10e6, limit_multiple=100000)
# run(10000, profile=True)
# run(1e9, limit_hours=11)
#
# def py_cb(cur_tstep):
#     print "callback called at step: ", str(cur_tstep)
#
# run(10000, callback_period=100, callback=py_cb)
# \endcode
#
# \b Overview
#
# Execute the run() command to advance the simulation forward in time.
# During the run, all previously specified \ref analyze "analyzers",
# \ref dump "dumps", \ref update "updaters" and the \ref integrate "integrators"
# are executed at the specified regular periods.
#
# After run() completes, you may change parameters of the simulation
# and continue the simulation by executing run() again. Time steps are added
# cumulatively, so calling run(1000) and then run(2000) would run the simulation
# up to time step 3000.
#
# run() cannot be executed before the system is \ref init "initialized". In most
# cases, run() should only be called after after pair forces, bond forces,
# and an \ref integrate "integrator" are specified.
#
# When \a profile is \em True, a detailed breakdown of how much time was spent in each
# portion of the calculation is printed at the end of the run. Collecting this timing information
# can slow the simulation significantly.
#
# <b>Wallclock limited runs</b>
#
# There are a number of mechanisms to limit the time of a running hoomd script. Use these in a job
# queuing environment to allow your script to cleanly exit before reaching the system enforced walltime limit.
#
# Force run() to end only on time steps that are a multiple of \a limit_mulitple. Set this to the period at which you
# dump restart files so that you always end a run() cleanly at a point where you can restart from. Use
# \a phase=0 on logs, file dumps, and other periodic tasks. With phase=0, these tasks will continue on the same
# sequence regardless of the restart period.
#
# Set the environment variable `HOOMD_WALLTIME_STOP` prior to executing `hoomd` to stop the run() at a given wall
# clock time. run() monitors performance and tries to ensure that it will end *before* `HOOMD_WALLTIME_STOP`. This
# environment variable works even with multiple stages of runs in a script (use run_upto()). Set the variable to
# a unix epoch time. For example in a job script that should run 12 hours, set `HOOMD_WALLTIME_STOP` to 12 hours from
# now, minus 10 minutes to allow for job cleanup.
# ~~~
# export HOOMD_WALLTIME_STOP=$((`date +%s` + 12 * 3600 - 10 * 60))
# ~~~
#
# When using `HOOMD_WALLTIME_STOP`, run() will throw the exception `WalltimeLimitReached` if it exits due to the walltime
# limit. For more information on using this exception, see (TODO: page to be written).#
#
# \a limit_hours is another way to limit the length of a run(). Set it to a number of hours (use fractional values for
# minutes) to limit this particular run() to that length of time. This is less useful than `HOOMD_WALLTIME_STOP` in a
# job queuing environment.
#
# \b Callbacks
#
# If \a callback is set to a Python function then this function will be called regularly
# at \a callback_period intervals. The callback function must receive one integer as argument
# and can return an integer. The argument passed to the callback is the current time step number.
# If the callback function returns a negative number, the run is immediately aborted.
#
# If \a callback_period is set to 0 (the default) then the callback is only called
# once at the end of the run. Otherwise the callback is executed whenever the current
# time step number is a multiple of \a callback_period.
#
def run(tsteps, profile=False, limit_hours=None, limit_multiple=1, callback_period=0, callback=None, quiet=False):
    if not quiet:
        _util.print_status_line();
    # check if initialization has occured
    if not init.is_initialized():
        context.msg.error("Cannot run before initialization\n");
        raise RuntimeError('Error running');

    if globals.integrator is None:
        context.msg.warning("Starting a run without an integrator set");
    else:
        globals.integrator.update_forces();
        globals.integrator.update_methods();
        globals.integrator.update_thermos();

    # update autotuner parameters
    globals.system.setAutotunerParams(context.options.autotuner_enable, int(context.options.autotuner_period));

    # if rigid bodies, setxv
    if len(data.system_data(globals.system_definition).bodies) > 0:
        data.system_data(globals.system_definition).bodies.updateRV()

    for logger in globals.loggers:
        logger.update_quantities();
    globals.system.enableProfiler(profile);
    globals.system.enableQuietRun(quiet);

    if globals.neighbor_list:
        globals.neighbor_list.update_rcut();
        globals.neighbor_list.update_exclusions_defaults();

    # update all user-defined neighbor lists
    for nl in globals.neighbor_lists:
        nl.update_rcut()
        nl.update_exclusions_defaults()

    # detect 0 hours remaining properly
    if limit_hours == 0.0:
        context.msg.warning("Requesting a run() with a 0 time limit, doing nothing.\n");
        return;
    if limit_hours is None:
        limit_hours = 0.0

    if not quiet:
        context.msg.notice(1, "** starting run **\n");
    globals.system.run(int(tsteps), callback_period, callback, limit_hours, int(limit_multiple));
    if not quiet:
        context.msg.notice(1, "** run complete **\n");

## \brief Runs the simulation up to a given time step number
#
# \param step Final time step of the simulation which to run
# \param keywords (see below) Catch for all keyword arguments to pass on to run()
#
# run_upto() runs the simulation, but only until it reaches the given time step, \a step. If the simulation has already
# reached the specified step, a message is printed and no simulation steps are run.
#
# It accepts all keyword options that run() does.
#
# \b Examples:
# \code
# run_upto(1000)
# run_upto(10000, profile=True)
# run_upto(1e9, limit_hours=11)
# \endcode
#
def run_upto(step, **keywords):
    if 'quiet' in keywords and not keywords['quiet']:
        _util.print_status_line();
    # check if initialization has occured
    if not init.is_initialized():
        context.msg.error("Cannot run before initialization\n");
        raise RuntimeError('Error running');

    # determine the number of steps to run
    step = int(step);
    cur_step = globals.system.getCurrentTimeStep();

    if cur_step >= step:
        context.msg.notice(2, "Requesting run up to a time step that has already passed, doing nothing\n");
        return;

    n_steps = step - cur_step;

    _util._disable_status_lines = True;
    run(n_steps, **keywords);
    _util._disable_status_lines = False;

## Get the current simulation time step
#
# \returns current simulation time step
def get_step():
    # check if initialization has occurred
    if not init.is_initialized():
        context.msg.error("Cannot get step before initialization\n");
        raise RuntimeError('Error getting step');

    return globals.system.getCurrentTimeStep();

## Start CUDA profiling
#
# When using nvvp to profile CUDA kernels in hoomd jobs, you usually don't care about all the initialization and
# startup. cuda_profile_start() allows you to not even record that. To use, uncheck the box "start profiling on
# application start" in your nvvp session configuration. Then, call cuda_profile_start() in your hoomd script when
# you want nvvp to start collecting information.
#
# Example:
# ~~~~~
# from hoomd_script import *
# init.read_xml("init.xml");
# # setup....
# run(30000);  # warm up and auto-tune kernel block sizes
# option.set_autotuner_params(enable=False);  # prevent block sizes from further autotuning
# cuda_profile_start();
# run(100);
# ~~~~~
def cuda_profile_start():
    hoomd.cuda_profile_start();

## Stop CUDA profiling
# \sa cuda_profile_start();
def cuda_profile_stop():
    hoomd.cuda_profile_stop();

# Check to see if we are built without MPI support and the user used mpirun
if (not hoomd.is_MPI_available()) and ('OMPI_COMM_WORLD_RANK' in os.environ or 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ):
    print('HOOMD-blue is built without MPI support, but seems to have been launched with mpirun');
    print('exiting now to prevent many sequential jobs from starting');
    raise RuntimeError('Error launching hoomd')
