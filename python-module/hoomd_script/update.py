# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

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

# Maintainer: joaander / All Developers are free to add commands for new features

import hoomd;
import globals;
import compute;
import util;
import variant;
import sys;
import init;

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
        # check if initialization has occurred
        if not init.is_initialized():
            globals.msg.error("Cannot create updater before initialization\n");
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
            globals.msg.error("I don't know what to do with a period of type " + str(type(period)) + "expecting an int or a function\n");
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
    # \brief Saved period retrieved when an updater is disabled: used to set the period when re-enabled

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_updater is None:
            globals.msg.error('Bug in hoomd_script: cpp_updater not set, please report\n');
            raise RuntimeError();

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
        self.check_initialization();
        
        # check if we are already disabled
        if not self.enabled:
            globals.msg.warning("Ignoring command to disable an updater that is already disabled");
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
        self.check_initialization();
            
        # check if we are already disabled
        if self.enabled:
            globals.msg.warning("Ignoring command to enable an updater that is already enabled");
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
            globals.msg.warning("A period cannot be changed to a variable one");
        else:
            globals.msg.warning("I don't know what to do with a period of type " + str(type(period)) + " expecting an int or a function");

# **************************************************************************

## Sorts particles in memory to improve cache coherency
#
# Every \a period time steps, particles are reordered in memory based on
# a Hilbert curve. This operation is very efficient, and the reordered particles
# significantly improve performance of all other algorithmic steps in HOOMD. 
# 
# The reordering is accomplished by placing particles in spatial bins. A Hilbert curve is generated that traverses
# these bins and particles are reordered in memory in the same order in which 
# they fall on the curve. The grid dimension used over the course of the simulation is held constant, and the default
# is chosen to be as fine as possible without utilizing too much memory. The dimension can be changed with set_params(),
# just be aware that the value chosen will be rounded up to the next power of 2 and that the amount of memory usage for
# 3D simulations grows very quickly:
# - \a grid=128 uses 8 MB
# - \a grid=256 uses 64 MB
# - \a grid=512 uses 512 MB
# - \a grid=1024 uses 4096 MB
#
# 2D simulations do not use any additional memory and default to \a grid=4096
# 
# Because all simulations benefit from this process, a sorter is created by 
# default. If you have reason to disable it or modify parameters, you
# can use the built-in variable \c sorter to do so after initialization. The
# following code example disables the sorter. The init.create_random command
# is just an example; sorter can be modified after any command that initializes 
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
    # By default, the sorter is created with a \a grid of 256 (4096 in 2D) and
    # an update period of 300 time steps (100 if running on the CPU).
    # The period can be changed with set_period() and the grid width can be
    # changed with set_params()
    def __init__(self):
        # initialize base class
        _updater.__init__(self);
        
        # create the c++ mirror class
        self.cpp_updater = hoomd.SFCPackUpdater(globals.system_definition);
        
        default_period = 300;
        # change default period to 100 on the CPU
        if not globals.exec_conf.isCUDAEnabled():
            default_period = 100;
            
        self.setupUpdater(default_period);

    ## Change sorter parameters
    #
    # \param grid New grid dimension (if set)
    # 
    # \b Examples:
    # \code
    # sorter.set_params(grid=128)
    # \endcode
    def set_params(self, grid=None):
        util.print_status_line();
        self.check_initialization();
        
        if grid is not None:
            self.cpp_updater.setGrid(grid);


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
    # \param T Temperature set point (in energy units)
    # \param period Velocities will be rescaled every \a period time steps
    #
    # \a T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # \b Examples:
    # \code
    # update.rescale_temp(T=1.2)
    # rescaler = update.rescale_temp(T=0.5)
    # update.rescale_temp(period=100, T=1.03)
    # update.rescale_temp(period=100, T=variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, T, period=1):
        util.print_status_line();
    
        # initialize base class
        _updater.__init__(self);
        
        # setup the variant inputs
        T = variant._setup_variant_input(T);
        
        # create the compute thermo
        thermo = compute._get_unique_thermo(group=globals.group_all);
        
        # create the c++ mirror class
        self.cpp_updater = hoomd.TempRescaleUpdater(globals.system_definition, thermo.cpp_compute, T.cpp_variant);
        self.setupUpdater(period);

    ## Change rescale_temp parameters
    #
    # \param T New temperature set point (in energy units)
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
        self.check_initialization();
            
        if T is not None:
            T = variant._setup_variant_input(T);
            self.cpp_updater.setT(T.cpp_variant);

## Zeroes system momentum
#
# Every \a period time steps, particle velocities are modified such that the total linear 
# momentum of the system is set to zero.
#
# update.zero_momentum is intended to be used when the \ref integrate.nve "NVE" integrator has the
# \a limit option specified, where Newton's third law is broken and systems could gain momentum.
# Of course, it can be used in any script.
#
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

## Enforces 2D simulation
#
# Every time step, particle velocities and accelerations are modified so that their z components are 0: forcing
# 2D simulations when other calculations may cause particles to drift out of the plane.
#
# Using enforce2d is only allowed when the system is specified as having only 2 dimensions. This specification can
# be made in the xml file read by init.read_xml() or set dynamically via the particle data access routines. Setting
# the number of dimensions to 2 also changes the degrees of freedom calculation for temperature calculations and forces
# the neighbor list to only find 2D neighbors. Doing so requires that a small, but non-zero, value be set for the z
# dimension of the simulation box.
#
class enforce2d(_updater):
    ## Initialize the 2D enforcement
    #
    # \b Examples:
    # \code
    # update.enforce2d()
    # \endcode
    #
    def __init__(self):
        util.print_status_line();
        period = 1;
    
        # initialize base class
        _updater.__init__(self);
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_updater = hoomd.Enforce2DUpdater(globals.system_definition);
        else:
            self.cpp_updater = hoomd.Enforce2DUpdaterGPU(globals.system_definition);
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
    # \param Lx the value of the box length in the x direction as a function of time (in distance units)
    # \param Ly (if set) the value of the box length in the y direction as a function of time (in distance units)
    # \param Lz (if set) the value of the box length in the z direction as a function of time (in distance units)
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
        if Ly is None:
            Ly = Lx;
        if Lz is None:
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
    # box_resize.set_params(scale_particles = True)
    # \endcode
    def set_params(self, scale_particles=None):
        util.print_status_line();
        self.check_initialization();
            
        if scale_particles is not None:
            self.cpp_updater.setParams(scale_particles);

# Global current id counter to assign updaters unique names
_updater.cur_id = 0;

class remove_particles(_updater):
    ## Initialize the particle remover
    #
    # \param period This updater will be applied every period timesteps.
    # \param filename Removed particles will be output to this file with their trajectories.
    # \param groupA Primary group that will be checked for departing particles.
    # \param groupB Secondary group that will be checked for departing particles
    # \param i_ion Ion impact for this simulation.
    # \param ion_energy Impact energy of the ion.
    # \param ion_phi Azimuthal angle of the ion.
    # \param r_cut Cutoff radius for finding molecules.
    # \param overwrite Determines whether the exisiting output file is overwritten.
    #
    def __init__(self, groupA, groupB, i_ion, ion_energy, ion_phi, r_cut=2.1, period=1, filename="output/sputtered.txt", overwrite=False):
        util.print_status_line();

        # initialize the base class
        _updater.__init__(self);

        # create the compute thermo
        thermo = compute._get_unique_thermo(group = groupA);

        # update the neighbor list
        neighbor_list = pair._update_global_nlist(r_cut);
	# neighbor_list.subscribe(lambda: self.log*self.get_max_rcut());

	# shift ion_phi by 180 degrees
	ion_phi = ion_phi - 180.0;

	# output the ion information to the file
	if overwrite:
	    f = open(filename, "w")
	else:
	    f = open(filename, "a")
	f.write( '%d\t%lf\t%lf\n' % (i_ion, ion_energy, ion_phi) )
	f.close()

        # initialize the reflected c++ class
        self.cpp_updater = hoomd.RemoveParticlesUpdater(globals.system_definition,
                                                        groupA.cpp_group,
							groupB.cpp_group,
                                                        thermo.cpp_compute,
                                                        neighbor_list.cpp_nlist,
							r_cut,
                                                        filename);
        self.setupUpdater(period);
