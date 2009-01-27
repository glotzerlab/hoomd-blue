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

import globals;
import sys;
import hoomd;
import util;

## \package hoomd_script.force
# \brief Other types of forces
#
# This package contains various forces that don't belong in any of the other categories

## \internal
# \brief Base class for forces
#
# A force in hoomd_script reflects a ForceCompute in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ analyzer itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _force:
	## \internal
	# \brief Constructs the force
	#
	# Initializes the cpp_analyzer to None.
	# Assigns a name to the force in force_name;
	def __init__(self):
		# check if initialization has occured
		if globals.system == None:
			print >> sys.stderr, "\n***Error! Cannot create force before initialization\n";
			raise RuntimeError('Error creating force');
		
		self.cpp_force = None;

		# increment the id counter
		id = _force.cur_id;
		_force.cur_id += 1;
		
		self.force_name = "force%d" % (id);
		self.enabled = True;
		globals.forces.append(self);

	## \var enabled
	# \internal
	# \brief True if the force is enabled

	## \var cpp_force
	# \internal
	# \brief Stores the C++ side ForceCompute managed by this class
	
	## \var force_name
	# \internal
	# \brief The Force's name as it is assigned to the System

	## Disables the force
	#
	# \b Examples:
	# \code
	# force.disable()
	# \endcode
	#
	# Executing the disable command will remove the force from the simulation.
	# Any run() command executed after disabling a force will not calculate or 
	# use the force during the simulation. A disabled force can be re-enabled
	# with enable()
	#
	# To use this command, you must have saved the force in a variable, as 
	# shown in this example:
	# \code
	# force = pair.some_force()
	# # ... later in the script
	# force.disable()
	# \endcode
	def disable(self):
		util.print_status_line();
		
		# check that we have been initialized properly
		if self.cpp_force == None:
			print >> sys.stderr, "\nBug in hoomd_script: cpp_force not set, please report\n";
			raise RuntimeError('Error disabling force');
			
		# check if we are already disabled
		if not self.enabled:
			print "***Warning! Ignoring command to disable a force that is already disabled";
			return;
		
		globals.system.removeCompute(self.force_name);
		self.enabled = False;
		globals.forces.remove(self);

	## Enables the force
	#
	# \b Examples:
	# \code
	# force.enable()
	# \endcode
	#
	# See disable() for a detailed description.
	def enable(self):
		util.print_status_line();
		
		# check that we have been initialized properly
		if self.cpp_force == None:
			print >> sys.stderr, "\nBug in hoomd_script: cpp_force not set, please report\n";
			raise RuntimeError('Error enabling force');
			
		# check if we are already disabled
		if self.enabled:
			print "***Warning! Ignoring command to enable a force that is already enabled";
			return;
			
		globals.system.addCompute(self.cpp_force, self.force_name);
		self.enabled = True;
		globals.forces.append(self);
		
	## \internal
	# \brief updates force coefficients
	def update_coeffs(self):
		pass
		raise RuntimeError("_force.update_coeffs should not be called");
		# does nothing: this is for derived classes to implement
	
# set default counter
_force.cur_id = 0;


## Constant %force
#
# The command force.constant specifies that a %constant %force should be added to every
# particle in the simulation.
#
class constant(_force):
	## Specify the %constant %force
	#
	# \param fx x-component of the %force
	# \param fy y-component of the %force
	# \param fz z-component of the %force
	#
	# \b Examples:
	# \code
	# force.constant(fx=1.0, fy=0.5, fz=0.25)
	# const = force.constant(fx=0.4, fy=1.0, fz=0.5)
	# \endcode
	def __init__(self, fx, fy, fz):
		util.print_status_line();
		
		# initialize the base class
		_force.__init__(self);
		
		# create the c++ mirror class
		self.cpp_force = hoomd.ConstForceCompute(globals.system_definition, fx, fy, fz);
			
		globals.system.addCompute(self.cpp_force, self.force_name);
		
	## Change the value of the force
	#
	# \param fx New x-component of the %force
	# \param fy New y-component of the %force
	# \param fz New z-component of the %force
	#
	# Using set_force() requires that you saved the created %constant %force in a variable. i.e.
	# \code
	# const = force.constant(fx=0.4, fy=1.0, fz=0.5)
	# \endcode
	#
	# \b Example:
	# \code
	# const.set_force(fx=0.2, fy=0.1, fz=-0.5)
	# \endcode
	def set_force(self, fx, fy, fz):
		# check that we have been initialized properly
		if self.cpp_force == None:
			print >> sys.stderr, "\nBug in hoomd_script: cpp_force not set, please report\n";
			raise RuntimeError('Error enabling force');
			
		self.cpp_force.setForce(fx, fy, fz);
		
	# there are no coeffs to update in the constant force compute
	def update_coeffs(self):
		pass

