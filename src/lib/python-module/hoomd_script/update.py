# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$

import hoomd;
import globals;

## \package hoomd_script.update
# \brief Commands that modify the system state in some way
#
# Details

## \internal
# \brief Base class for updaters
#
# An updater in hoomd_script reflects an Updater in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ upater itself is tracked and added to the
# System 2) methods are provided for disabling the updater and changing the 
# period which the system calls it
class _updater:
	## \internal
	# \brief Constructs the updater
	#
	# \param self Python-required class instance variable
	#
	# Initializes the cpp_updater to None.
	# Assigns a name to the updater in updater_name;
	def __init__(self):
		# check if initialization has occured
		if globals.system == None:
			print "Error: Cannot create updater before initialization";
			raise RuntimeError('Error creating updater');
		
		self.cpp_updater = None;

		# increment the id counter
		id = _updater.cur_id;
		_updater.cur_id += 1;
		
		self.updater_name = "updater%d" % (id);
		self.enabled = True;

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
	# \param self Python-required class instance variable
	#
	# \b Examples:<br>
	# updater.disable()
	#
	# Executing the disable command will remove the updater from the system.
	# Any run() command exected after disabling an updater will not use that 
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
		print "updater.disable()";
		
		# check that we have been initialized properly
		if self.cpp_updater == None:
			"Bug in hoomd_script: cpp_updater not set, please report";
			raise RuntimeError('Error disabling updater');
			
		# check if we are already disabled
		if not self.enabled:
			print "Warning: Ignoring command to disable an updater that is already disabled";
			return;
		
		self.prev_period = globals.system.getUpdaterPeriod(self.updater_name);
		globals.system.removeUpdater(self.updater_name);
		self.enabled = False;

	## Enables the updater
	#
	# \param self Python-required class instance variable
	#
	# \b Examples:<br>
	# updater.enable()
	#
	# See disable() for a detailed description.
	def enable(self):
		print "updater.enable()";
		
		# check that we have been initialized properly
		if self.cpp_updater == None:
			"Bug in hoomd_script: cpp_updater not set, please report";
			raise RuntimeError('Error enabling updater');
			
		# check if we are already disabled
		if self.enabled:
			print "Warning: Ignoring command to enable an updater that is already enabled";
			return;
			
		globals.system.addUpdater(self.cpp_updater, self.updater_name, self.prev_period);
		self.enabled = True;
		
	## Changes the period between updater executions
	#
	# \param self Python-required class instance variable
	# \param period New period to set
	#
	# \b Examples:<br>
	# updater.set_period(100);<br>
	# updater.set_period(1);<br>
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
		print "updater.set_period(", period, ")";
		globals.system.setUpdaterPeriod(self.updater_name, period);

# **************************************************************************

## Sorts particles in memory to improve cache coherency
#
# Every \a period time steps, particles are reordered in memory based on
# a Hilbert curve. This operation is very efficient, and the reordered particles
# significanly improve performance of all other algorithmic steps in HOOMD. 
# 
# The reordering is accomplished by placing particles in spatial bins
# \a bin_width distance units wide. A Hilbert curve is generated that traverses
# these bins and particles are reorderd in memory in the same order in which 
# they fall on the curve. Testing indicates that a bin width equal to the
# particle diameter works well, though it may lead to excessive memory usage
# in extremely low density systems. set_params() can be used to increase the
# bin width in such situations.
# 
# Because all simulations benefit from this process, a sorter is created by 
# default. If you have reason to disable it or modify parameters, you
# can use the built-in variable sorter to do so after initialization. The
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
	# \param self Python-required class instance variable
	# 
	# Users should not initialize the sorter directly. One in created for you
	# when any init.create_* or init.read_* commands are issued to initialize the 
	# system. The created sorter can be accessed via the built-in variable sorter.
	#
	# By default, the sorter is created with a \a bin_width of 1.0 and
	# an update period of 500 time steps. The period can be changed with
	# set_period() and the bin width can be changed with set_params()
	def __init__(self):
		# initialize base class
		_updater.__init__(self);
		
		# create the c++ mirror class
		self.cpp_updater = hoomd.SFCPackUpdater(globals.particle_data, 1.0);
		globals.system.addUpdater(self.cpp_updater, self.updater_name, 500);

	## Change sorter parameters
	#
	# \param self Python-required class instance variable
	# \param bin_width New bin width (if set)
	# 
	# \b Examples:<br>
	# sorter.set_params(bin_width=2.0)
	def set_params(self, bin_width=None):
		print "sorter.set_params(bin_width =", bin_width, ")";
	
		# check that proper initialization has occured
		if self.cpp_updater == None:
			print "Bug in hoomd_script: cpp_updater not set, please report";
			raise RuntimeError('Error setting sorter parameters');
		
		if bin_width != None:
			self.cpp_updater.setBinWidth(bin_width);


## Rescales particle velocities
#
# Every \a period time steps, particle velocities are rescaled by equal factors
# so that they are consistent with a given temperature in the equipartition theorem
# \f$\langle 1/2 m v^2 \rangle = k_B T \f$. 
#
# rescale_temp is best coupled with the \ref integrate.nve "NVE" integrator.
class rescale_temp(_updater):
	## Initialize the rescaler
	#
	# \param self Python-required class instance variable
	# \param T Temperature set point
	# \param period Velocities will be rescaled every \a period time steps
	# 
	# \b Examples:<br>
	# update.rescale_temp(T=1.2)<br>
	# rescaler = update.rescale_temp(T=0.5)<br>
	# update.rescale_temp(period=100, T=1.03)<br>
	def __init__(self, T, period=1):
		print "update.rescale_temp(T =", T, ")";
	
		# initialize base class
		_updater.__init__(self);
		
		# create the c++ mirror class
		self.cpp_updater = hoomd.TempRescaleUpdater(globals.particle_data, hoomd.TempCompute(globals.particle_data), T);
		globals.system.addUpdater(self.cpp_updater, self.updater_name, period);

	## Change rescale_temp parameters
	#
	# \param self Python-required class instance variable
	# \param T New temperature set point
	# 
	# \b Examples:<br>
	# rescaler.set_params(T=2.0)
	def set_params(self, T=None):
		print "rescaler.set_params(T=", T, ")";
	
		# check that proper initialization has occured
		if self.cpp_updater == None:
			print "Bug in hoomd_script: cpp_updater not set, please report";
			raise RuntimeError('Error setting temp_rescale parameters');
			
		if T != None:
			self.cpp_updater.setT(T);



# Global current id counter to assign updaters unique names
_updater.cur_id = 0;