# -*- coding: iso-8859-1 -*-
#Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
#(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
#Iowa State University and The Regents of the University of Michigan All rights
#reserved.

#HOOMD-blue may contain modifications ("Contributions") provided, and to which
#copyright is held, by various Contributors who have granted The Regents of the
#University of Michigan the right to modify and/or distribute such Contributions.

#Redistribution and use of HOOMD-blue, in source and binary forms, with or
#without modification, are permitted, provided that the following conditions are
#met:

#* Redistributions of source code must retain the above copyright notice, this
#list of conditions, and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice, this
#list of conditions, and the following disclaimer in the documentation and/or
#other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of HOOMD-blue's
#contributors may be used to endorse or promote products derived from this
#software without specific prior written permission.

#Disclaimer

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
#ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$
# Maintainer: joaander

import hoomd;
import sys;
import util;

## \package hoomd_script
# \brief Base module for the user-level scripting API
# 
# hoomd_script provides a very high level user interface for executing 
# simulations using HOOMD. This python module is designed to be imported
# into python with "from hoomd_script import *"

## \internal
# \brief Internal python variable 
__all__ = [ "analyze", 
            "bond", 
            "angle", 
            "dihedral", 
            "improper", 
            "dump", 
            "force", 
            "globals", 
            "group", 
            "init", 
            "integrate", 
            "pair", 
            "update", 
            "wall",
            "variant", 
            "run", 
            "tune", 
            "hoomd",
            "get_hoomd_script_version"];
            
## \internal
# \brief Major version of hoomd_script
version_major = 0;

## \internal
# \brief Minor version of hoomd_script
version_minor = 2;

## \brief Get the version information of hoomd_script
# \returns a tuple (major, minor)
#
# This version is the version number of \b hoomd_script , not HOOMD as a whole. It is intended for use by
# third party API plugins that interface with hoomd_script. When new features are added (i.e. a new command
# or a new option to an existing command), the minor version will be incremented. When major changes are implemented
# or changes that break backwards compatibility are made, then the major version is incremented and the minor reset
# to 0. Only one such increment of either type will occur per each tagged release of HOOMD.
def get_hoomd_script_version():
    return (version_major, version_minor)

## \brief Runs the simulation for a given number of time steps
#
# \param tsteps Number of time steps to advance the simulation by
# \param profile Set to True to enable detailed profiling
# \param limit_hours  (if set) Limit the run to a given number of hours.
# \param callback     (if set) Sets a Python function to be called regularly during a run.
# \param callback_period Sets the period, in time steps, between calls made to \a callback
# \param quiet Set to True to eliminate the status information printed to the screen by the run
#
# \b Examples:
# \code
# run(1000)
# run(10e6)
# run(10000, profile=True)
# run(1e9, limit_hours=11)
#
# def py_cb(cur_tstep):
#     print "callback called at step: ", str(cur_tstep)
#
# run(10000, callback_period=100, callback=py_cb)
# \endcode
#
# Execute the run() command to advance the simulation forward in time. 
# During the run, all previously specified \ref analyze "analyzers", 
# \ref dump "dumps", \ref update "updaters" and the \ref integrate "integrators"
# are executed at the specified regular periods.
# 
# After run() completes, you may change parameters of the simulation (i.e. temperature)
# and continue the simulation by executing run() again. Time steps are added
# cumulatively, so calling run(1000) and then run(2000) would run the simulation
# up to time step 3000.
#
# run() cannot be executed before the system is \ref init "initialized". In most 
# cases, it also doesn't make sense to execute run() until after pair forces, bond forces,
# and an \ref integrate "integrator" have been created.
#
# When \a profile is \em True, a detailed breakdown of how much time was spent in each
# portion of the calculation is printed at the end of the run. Collecting this timing information
# can slow the simulation on the GPU significantly, so only enable profiling for testing
# and troubleshooting purposes.
#
# If \a limit_hours is changed from the default of None, the run will continue until either
# the specified number of time steps has been reached, or the given number of hours has
# elapsed. This option can be useful in shared machines where the queuing system limits
# job run times. A fractional value can be given to limit a run to only a few minutes,
# if needed.
#
# If \a callback is set to a Python function then this function will be called regularly
# at \a callback_period intervals. The callback function must receive one integer as argument
# and can return an integer. The argument is the current time step number,
# and if the callback function returns a negative number then the run is immediately aborted.
# all other return values are currently ignored.
#
# If \a callback_period is set to 0 (the default) then the callback is only called
# once at the end of the run. Otherwise the callback is executed whenever the current
# time step number is a multiple of \a callback_period.
#
def run(tsteps, profile=False, limit_hours=None, callback_period=0, callback=None, quiet=False):
    if not quiet:
        util.print_status_line();
    # check if initialization has occured
    if (globals.system == None):
        print >> sys.stderr, "\n***Error! Cannot run before initialization\n";
        raise RuntimeError('Error running');
        
    if (globals.integrator == None):
        print "***Warning! Starting a run without an integrator set";
    else:
        globals.integrator.update_forces();
        globals.integrator.update_methods();
    
    for logger in globals.loggers:
        logger.update_quantities();
    globals.system.enableProfiler(profile);
    globals.system.enableQuietRun(quiet);
    
    if globals.neighbor_list:
        globals.neighbor_list.update_rcut();
        globals.neighbor_list.update_exclusions_defaults();

    if limit_hours == None:
        limit_hours = 0.0
    
    if not quiet:
        print "** starting run **"
    globals.system.run(int(tsteps), callback_period, callback, limit_hours);
    if not quiet:
        print "** run complete **"

