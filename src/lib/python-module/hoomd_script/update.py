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
# Maintainer: joaander / All Developers are free to add commands for new features

import hoomd;
import globals;
import util;
import variant;
import sys;

## \package hoomd_script.update
# \brief Commands that modify the system state in some way
#
# When an updater is specified, it acts on the particle system each time step to change
# it in some way. See the documentation of specific updaters to find out what they do.

## \internal
# \brief Base class for updaters
#
# An updater in hoomd_script reflects an Updater in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ updater itself is tracked and added to the
# System 2) methods are provided for disabling the updater and changing the 
# period which the system calls it
class _updater:
    ## \internal
    # \brief Constructs the updater
    #
    # Initializes the cpp_updater to None.
    # Assigns a name to the updater in updater_name;
    def __init__(self):
        # check if initialization has occured
        if globals.system == None:
            print >> sys.stderr, "\n***Error! Cannot create updater before initialization\n";
            raise RuntimeError('Error creating updater');
        
        self.cpp_updater = None;

        # increment the id counter
        id = _updater.cur_id;
        _updater.cur_id += 1;
        
        self.updater_name = "updater%d" % (id);
        self.enabled = True;
        
    ## \internal
    # 
    # \brief Helper function to setup updater period
    #
    # \param period An integer or callable function period
    #
    # If an integer is specified, then that is set as the period for the analyzer.
    # If a callable is passed in as a period, then a default period of 1000 is set 
    # to the integer period and the variable period is enabled
    #
    def setupUpdater(self, period):
        if type(period) == type(1.0):
            period = int(period);
        
        if type(period) == type(1):
            globals.system.addUpdater(self.cpp_updater, self.updater_name, period);
        elif type(period) == type(lambda n: n*2):
            globals.system.addUpdater(self.cpp_updater, self.updater_name, 1000);
            globals.system.setUpdaterPeriodVariable(self.updater_name, period);
        else:
            print >> sys.stderr, "\n***Error! I don't know what to do with a period of type", type(period), "expecting an int or a function\n";
            raise RuntimeError('Error creating updater');

    ## \var enabled
    # \internal
    # \brief True if the updater is enabled

    ## \var cpp_updater
    # \internal
    # \brief Stores the C++ side Updater managed by this class
    
    ## \var updater_name
    # \internal
    # \brief The Updater's name as it is assigned to the System

    ## \var prev_period
    # \internal
    # \brief Saved period retrived when an updater is disabled: used to set the period when re-enabled

    ## Disables the updater
    #
    # \b Examples:
    # \code
    # updater.disable()
    # \endcode
    #
    # Executing the disable command will remove the updater from the system.
    # Any run() command executed after disabling an updater will not use that 
    # updater during the simulation. A disabled updater can be re-enabled
    # with enable()
    #
    # To use this command, you must have saved the updater in a variable, as 
    # shown in this example:
    # \code
    # updater = update.some_updater()
    # # ... later in the script
    # updater.disable()
    # \endcode
    def disable(self):
        util.print_status_line();
        
        # check that we have been initialized properly
        if self.cpp_updater == None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_updater not set, please report\n";
            raise RuntimeError('Error disabling updater');
            
        # check if we are already disabled
        if not self.enabled:
            print "***Warning! Ignoring command to disable an updater that is already disabled";
            return;
        
        self.prev_period = globals.system.getUpdaterPeriod(self.updater_name);
        globals.system.removeUpdater(self.updater_name);
        self.enabled = False;

    ## Enables the updater
    #
    # \b Examples:
    # \code
    # updater.enable()
    # \endcode
    #
    # See disable() for a detailed description.
    def enable(self):
        util.print_status_line();
        
        # check that we have been initialized properly
        if self.cpp_updater == None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_updater not set, please report\n";
            raise RuntimeError('Error enabling updater');
            
        # check if we are already disabled
        if self.enabled:
            print "***Warning! Ignoring command to enable an updater that is already enabled";
            return;
            
        globals.system.addUpdater(self.cpp_updater, self.updater_name, self.prev_period);
        self.enabled = True;
        
    ## Changes the period between updater executions
    #
    # \param period New period to set
    #
    # \b Examples:
    # \code
    # updater.set_period(100);
    # updater.set_period(1);
    # \endcode
    #
    # While the simulation is \ref run() "running", the action of each updater
    # is executed every \a period time steps.
    #
    # To use this command, you must have saved the updater in a variable, as 
    # shown in this example:
    # \code
    # updater = update.some_updater()
    # # ... later in the script
    # updater.set_period(10)
    # \endcode
    def set_period(self, period):
        util.print_status_line();
        
        if type(period) == type(1.0):
            period = int(period);
        
        if type(period) == type(1):
            if self.enabled:
                globals.system.setUpdaterPeriod(self.updater_name, period);
            else:
                self.prev_period = period;
        elif type(period) == type(lambda n: n*2):
            print "***Warning! A period cannot be changed to a variable one";
        else:
            print "***Warning! I don't know what to do with a period of type", type(period), "expecting an int or a function";

# **************************************************************************

## Sorts particles in memory to improve cache coherency
#
# Every \a period time steps, particles are reordered in memory based on
# a Hilbert curve. This operation is very efficient, and the reordered particles
# significantly improve performance of all other algorithmic steps in HOOMD. 
# 
# The reordering is accomplished by placing particles in spatial bins
# \a bin_width distance units wide. A Hilbert curve is generated that traverses
# these bins and particles are reordered in memory in the same order in which 
# they fall on the curve. Testing indicates that a bin width equal to the
# particle diameter works well, though it may lead to excessive memory usage
# in extremely low density systems. set_params() can be used to increase the
# bin width in such situations.
# 
# Because all simulations benefit from this process, a sorter is created by 
# default. If you have reason to disable it or modify parameters, you
# can use the built-in variable \c sorter to do so after initialization. The
# following code example disables the sorter. The init.create_random command
# is just an example, sorter can be modified after any command that initializes 
# the system.
# \code
# init.create_random(N=1000, phi_p=0.2)
# sorter.disable()
# \endcode
class sort(_updater):
    ## Initialize the sorter
    #
    # Users should not initialize the sorter directly. One in created for you
    # when any initialization command from init is run. 
    # The created sorter can be accessed via the built-in variable \c sorter.
    #
    # By default, the sorter is created with a \a bin_width of 1.0 and
    # an update period of 500 time steps (100 if running on the CPU).
    # The period can be changed with set_period() and the bin width can be
    # changed with set_params()
    def __init__(self):
        # initialize base class
        _updater.__init__(self);
        
        # create the c++ mirror class
        self.cpp_updater = hoomd.SFCPackUpdater(globals.system_definition, 1.0);
        
        default_period = 500;
        # change default period to 100 on the CPU
        if globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
            default_period = 100;
            
        self.setupUpdater(default_period);

    ## Change sorter parameters
    #
    # \param bin_width New bin width (if set)
    # 
    # \b Examples:
    # \code
    # sorter.set_params(bin_width=2.0)
    # \endcode
    def set_params(self, bin_width=None):
        util.print_status_line();
    
        # check that proper initialization has occured
        if self.cpp_updater == None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_updater not set, please report\n";
            raise RuntimeError('Error setting sorter parameters');
        
        if bin_width != None:
            self.cpp_updater.setBinWidth(bin_width);


## Rescales particle velocities
#
# Every \a period time steps, particle velocities are rescaled by equal factors
# so that they are consistent with a given temperature in the equipartition theorem
# \f$\langle 1/2 m v^2 \rangle = k_B T \f$. 
#
# update.rescale_temp is best coupled with the \ref integrate.nve "NVE" integrator.
class rescale_temp(_updater):
    ## Initialize the rescaler
    #
    # \param T Temperature set point
    # \param period Velocities will be rescaled every \a period time steps
    # 
    # \b Examples:
    # \code
    # update.rescale_temp(T=1.2)
    # rescaler = update.rescale_temp(T=0.5)
    # update.rescale_temp(period=100, T=1.03)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, T, period=1):
        util.print_status_line();
    
        # initialize base class
        _updater.__init__(self);
        
        # create the c++ mirror class
        self.cpp_updater = hoomd.TempRescaleUpdater(globals.system_definition, hoomd.TempCompute(globals.system_definition), T);
        self.setupUpdater(period);

    ## Change rescale_temp parameters
    #
    # \param T New temperature set point
    # 
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # rescaler = update.rescale_temp(T=0.5)
    # \endcode
    #
    # \b Examples:
    # \code
    # rescaler.set_params(T=2.0)
    # \endcode
    def set_params(self, T=None):
        util.print_status_line();
    
        # check that proper initialization has occured
        if self.cpp_updater == None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_updater not set, please report\n";
            raise RuntimeError('Error setting temp_rescale parameters');
            
        if T != None:
            self.cpp_updater.setT(T);

## Zeroes system momentum
#
# Every \a period time steps, particle velocities are modified such that the total linear 
# momentum of the system is set to zero.
#
# update.zero_momentum is intended to be used when the \ref integrate.nve "NVE" integrator has the
# \a limit option specified, where Newton's third law is broken and systems could gain momentum.
# However, nothing prevents update.zero_momentum from being used in any HOOMD script.
class zero_momentum(_updater):
    ## Initialize the momentum zeroer
    #
    # \param period Momentum will be zeroed every \a period time steps
    # 
    # \b Examples:
    # \code
    # update.zero_momentum()
    # zeroer= update.zero_momentum(period=10)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, period=1):
        util.print_status_line();
    
        # initialize base class
        _updater.__init__(self);
        
        # create the c++ mirror class
        self.cpp_updater = hoomd.ZeroMomentumUpdater(globals.system_definition);
        self.setupUpdater(period);
        
## Rescales the system box size
#
# Every \a period time steps, the system box size is updated to a value given by
# the user (in a variant). As an option, the particles can either be left in place
# as the box is changed or their positions can be scaled with the box.
#
class box_resize(_updater):
    ## Initialize box size resizer
    #
    # \param Lx the value of the box length in the x direction as a function of time
    # \param Ly (if set) the value of the box length in the y direction as a function of time
    # \param Lz (if set) the value of the box length in the z direction as a function of time
    # \param period The box size will be updated every \a period time steps
    # 
    # \a Lx, \a Ly, \a Lz can either be set to a constant number or a variant may be provided.
    #
    # \note If Ly or Lz (or both) are left as None, then they will be set to Lx as a convenience for 
    # defining cubes.
    #
    # \note
    # By default, particle positions are rescaled with the box. To change this behavior,
    # use set_params().
    # 
    # \b Examples:
    # \code
    # update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]))
    # box_resize = update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]), period = 10)
    # update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]), 
    #                   Ly = variant.linear_interp([(0, 20), (1e6, 60)]),
    #                   Lz = variant.linear_interp([(0, 10), (1e6, 80)]))
    # update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]), Ly = 10, Lz = 10)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, Lx, Ly = None, Lz = None, period = 1):
        util.print_status_line();
    
        # initialize base class
        _updater.__init__(self);
        
        # setup arguments
        if Ly == None:
            Ly = Lx;
        if Lz == None:
            Lz = Lx;
            
        Lx = variant._setup_variant_input(Lx);
        Ly = variant._setup_variant_input(Ly);
        Lz = variant._setup_variant_input(Lz);

        # create the c++ mirror class
        self.cpp_updater = hoomd.BoxResizeUpdater(globals.system_definition, Lx.cpp_variant, Ly.cpp_variant, Lz.cpp_variant);
        self.setupUpdater(period);
        
    ## Change box_resize parameters
    #
    # \param scale_particles Set to True to scale particles with the box. Set to False
    #        to have particles remain in place when the box is scaled.
    # 
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # box_resize = update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]), period = 10)
    # \endcode
    #
    # \b Examples:
    # \code
    # box_resize.set_params(scale_particles = False)
    # box_resize.set_params(scale_particles = Talse)
    # \endcode
    def set_params(self, scale_particles=None):
        util.print_status_line();
    
        # check that proper initialization has occured
        if self.cpp_updater == None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_updater not set, please report\n";
            raise RuntimeError('Error setting box_resize parameters');
            
        if scale_particles != None:
            self.cpp_updater.setParams(scale_particles);

# Global current id counter to assign updaters unique names
_updater.cur_id = 0;

